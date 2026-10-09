import argparse
from contextlib import contextmanager
import gzip
import multiprocessing
from pathlib import Path
import shutil
import tempfile


@contextmanager
def jax_trace(directory, name):
    import jax

    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        with jax.profiler.trace(temporary, create_perfetto_trace=True):
            yield
        traces = list(Path(temporary).rglob("perfetto_trace.json.gz"))
        if len(traces) != 1:
            raise RuntimeError(f"Expected one JAX Perfetto trace, found {len(traces)}")
        with gzip.open(traces[0], "rb") as source:
            with (directory / f"{name}.json").open("wb") as destination:
                shutil.copyfileobj(source, destination)


def torch_experiment(args, directory):
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from myconv import ConvModel

    if not torch.cuda.is_available():
        raise RuntimeError("cuda :(")

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = (
        ConvModel(
            args.size,
            args.size,
            args.channels,
            args.out_channels,
            args.filter,
            args.stride,
            args.padding,
        )
        .cuda()
        .eval()
    )

    with torch.no_grad():
        model.bias.zero_()

    x = torch.randn(args.batch, args.channels, args.size, args.size, device="cuda")
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with torch.no_grad():
        torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as build:
            with profile(activities=activities) as prof:
                label = "first_call" if args.variant == "eager" else "compilation"
                with record_function(label):
                    run = model
                    if args.variant == "inductor":
                        run = torch.compile(model, backend="inductor", dynamic=False)
                    elif args.variant == "cuda":
                        from torch.utils.cpp_extension import load

                        module = load(
                            name="myconv_profile",
                            sources=[str(Path(__file__).with_name("myconv_kernel.cu"))],
                            build_directory=build,
                            verbose=True,
                        )
                        run = lambda value: module.conv_cuda(
                            value, model.weight, args.stride, args.padding
                        )
                    output = run(x)
                    torch.cuda.synchronize()
            prof.export_chrome_trace(str(directory / "compilation.json"))
        reference = torch.nn.functional.conv2d(
            x, model.weight, model.bias, stride=args.stride, padding=args.padding
        )
        torch.testing.assert_close(output, reference, atol=1e-4, rtol=1e-4)
        for _ in range(args.warmup):
            output = run(x)
        torch.cuda.synchronize()
        with profile(activities=activities) as prof:
            for i in range(args.iterations):
                with record_function(f"iteration_{i}"):
                    output = run(x)
                    torch.cuda.synchronize()
        prof.export_chrome_trace(str(directory / "steady.json"))


def jax_experiment(args, directory):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from myconv_jax import conv2d_manual_jax

    devices = jax.devices("gpu")
    rng = np.random.default_rng(0)
    with jax.default_device(devices[0]):
        x = jnp.asarray(
            rng.standard_normal(
                (args.batch, args.channels, args.size, args.size)
            ).astype("float32")
        )
        w = jnp.asarray(
            rng.standard_normal(
                (args.out_channels, args.channels, args.filter, args.filter)
            ).astype("float32")
        )
        b = jnp.zeros(args.out_channels, dtype=jnp.float32)
    jax.block_until_ready((x, w, b))
    # Disable reduced-precision dot products for the comparison.
    jax.config.update("jax_default_matmul_precision", "highest")
    run = jax.jit(conv2d_manual_jax, static_argnames=("stride", "padding"))
    with jax_trace(directory, "compilation"):
        with jax.profiler.TraceAnnotation("host_compilation"):
            executable = run.lower(
                x, w, b, stride=args.stride, padding=args.padding
            ).compile()
    output = executable(x, w, b).block_until_ready()
    # Reference and host transfers are outside the measured traces.
    import torch

    reference = torch.nn.functional.conv2d(
        torch.from_numpy(np.asarray(x).copy()),
        torch.from_numpy(np.asarray(w).copy()),
        stride=args.stride,
        padding=args.padding,
    )
    np.testing.assert_allclose(
        np.asarray(output), reference.numpy(), atol=1e-4, rtol=1e-4
    )
    for _ in range(args.warmup):
        executable(x, w, b).block_until_ready()
    with jax_trace(directory, "steady"):
        for i in range(args.iterations):
            with jax.profiler.TraceAnnotation(f"iteration_{i}"):
                executable(x, w, b).block_until_ready()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--filters", type=int, nargs="+", default=[3, 5, 7])
    parser.set_defaults(
        batch=2,
        channels=3,
        out_channels=8,
        stride=1,
        padding=1,
        warmup=10,
        iterations=20,
        output=Path("profile_results").resolve(),
    )
    args = parser.parse_args()
    if min(args.sizes + args.filters) < 1:
        parser.error("Input and filter sizes must be positive.")
    if any(
        size + 2 * args.padding < kernel
        for size in args.sizes
        for kernel in args.filters
    ):
        parser.error("Filter is larger than the padded input.")

    context = multiprocessing.get_context("spawn")
    for variant in ("eager", "inductor", "cuda", "jax"):
        for size in args.sizes:
            for kernel in args.filters:
                print(f"Profiling {variant}: H=W={size}, K={kernel}", flush=True)
                process = context.Process(
                    target=experiment, args=(args, variant, size, kernel)
                )
                process.start()
                process.join()
                if process.exitcode:
                    raise SystemExit(process.exitcode)


def experiment(args, variant, size, kernel):
    args.variant, args.size, args.filter = variant, size, kernel
    tag = f"{args.variant}_N{args.batch}_C{args.channels}_H{args.size}_O{args.out_channels}_K{args.filter}_S{args.stride}_P{args.padding}"
    directory = args.output / tag
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("compilation.json", "steady.json")):
        raise RuntimeError(
            f"Experiment already exists: {directory}. Move the existing traces before rerunning."
        )

    if args.variant == "jax":
        jax_experiment(args, directory)
    else:
        torch_experiment(args, directory)
    print(f"Saved {directory / 'compilation.json'} and {directory / 'steady.json'}")


if __name__ == "__main__":
    main()
