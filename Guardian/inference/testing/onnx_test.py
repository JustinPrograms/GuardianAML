import os
import numpy as np
import onnxruntime as ort


def _shape(shape, batch: int = 1) -> tuple[int, ...]:
    out = []
    for d in shape:
        # ORT uses `None`/strings for dynamic dims.
        if d is None or isinstance(d, str):
            out.append(batch)
        else:
            out.append(int(d) if int(d) > 0 else batch)
    return tuple(out)


def main() -> int:
    model_path = "~/Guardian/datasets/engineered_testing_mini.csv"
    n = int(os.getenv("N", "10"))
    batch = int(os.getenv("BATCH", "1"))

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    print("Model:", model_path)
    print("Inputs:")
    for i in sess.get_inputs():
        print(" ", i.name, i.type, i.shape)
    print("Outputs:")
    for o in sess.get_outputs():
        print(" ", o.name, o.type, o.shape)

    # Basic pools; override via env if you want exact categories.
    string_pools = {
        "Payment_Format": os.getenv(
            "PAYMENT_FORMAT_POOL", "WIRE,ACH,SWIFT,CASH,CHECK,CARD,OTHER"
        ).split(","),
        "Receiving_Currency": os.getenv(
            "RECEIVING_CURRENCY_POOL", "USD,CAD,EUR,GBP,JPY"
        ).split(","),
        "Payment_Currency": os.getenv("PAYMENT_CURRENCY_POOL", "USD,CAD,EUR,GBP,JPY").split(","),
    }

    feeds = {}
    for i in sess.get_inputs():
        shp = _shape(i.shape, batch=batch)
        itype = (i.type or "").lower()

        if "string" in itype:
            pool = [s for s in string_pools.get(i.name, ["UNK"]) if s != ""]
            feeds[i.name] = np.random.choice(pool, size=shp).astype(object)
            continue

        # Default to float32 for numeric tensors.
        dt = np.float32
        if "float16" in itype:
            dt = np.float16
        elif "float" in itype:
            dt = np.float32
        elif "int64" in itype:
            dt = np.int64
        elif "int32" in itype:
            dt = np.int32

        if np.issubdtype(dt, np.floating):
            feeds[i.name] = np.random.rand(*shp).astype(dt)
        else:
            feeds[i.name] = np.random.randint(0, 10, size=shp, dtype=dt)

    for k, v in feeds.items():
        print("Feed", k, getattr(v, "dtype", type(v)), getattr(v, "shape", None))

    for t in range(n):
        outs = sess.run(None, feeds)
        print(f"Run {t}:")
        for name, arr in zip([o.name for o in sess.get_outputs()], outs):
            arr = np.asarray(arr)
            print(" ", name, arr.dtype, arr.shape, "sample=", arr.reshape(-1)[:5].tolist())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

