from os import name
from sys import flags
import tvm
from tvm import te
import numpy as np
import tvm.testing
from tvm.topi.cuda.tensor_intrin import (
    intrin_load_shared,
    intrin_wmma_load_matrix_A,
    intrin_wmma_load_matrix_W,
    intrin_wmma_gemm,
    intrin_wmma_store_matrix
)

@tvm.testing.requires_tensorcore
def test_tensor_core_gemm(shape, tb_tile_shape, warp_tile_shape, wmma_shape, dtype, mode="nn", flags = []):
    m, n, k = shape
    tm, tn, tk = tb_tile_shape
    wm, wn, wk = warp_tile_shape

    wmma_m, wmma_n, wmma_k = wmma_shape

    assert m % tm == 0
    assert n % tn == 0
    assert k % tk == 0

    assert tm % wm == 0
    assert tn % wn == 0
    assert tk % wk == 0

    assert wm % wmma_m == 0
    assert wn % wmma_n == 0
    assert wk % wmma_k == 0

    num_warp = (tm * tn) // (wm * wn)

    assert num_warp <= 32 
    
    Ah, Aw = m, k
    if mode == "nn": 
        Wh, Ww = k, n
    elif mode == "nt": 
        Wh, Ww = n, k
    else: 
        print("only `nn` and `nt` are supported")
        return -1

    A = te.placeholder((Ah, Aw), name="A", dtype=dtype)
    W = te.placeholder((Wh, Ww), name="W", dtype=dtype)

    dr = te.reduce_axis((0, k), "dr")

    if mode == "nn":
        B = te.compute((m, n), lambda h, w: te.sum(A[h, dr].astype("float32") * W[dr, w].astype("float32"), axis=dr), name="B")
    else:
        B = te.compute((m, n), lambda h, w: te.sum(A[h, dr].astype("float32") * W[w, dr].astype("float32"), axis=dr), name="B")
    s = te.create_schedule(B.op)

    # Memory hierarchy
    AS = s.cache_read(A, "shared", [B])
    WS = s.cache_read(W, "shared", [B])
    AF = s.cache_read(AS, "wmma.matrix_a", [B])
    WF = s.cache_read(WS, "wmma.matrix_b", [B])
    BF = s.cache_write(B, "wmma.accumulator")

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    # Thread Block Tile
    h, w = s[B].op.axis
    ho, hi = s[B].split(h, tm)
    wo, wi = s[B].split(w, tn)

    # Warp Tile
    hio, hii = s[B].split(hi, wm)
    wio, wii = s[B].split(wi, wn)

    # wmma tile
    hiio, hiii = s[B].split(hii, wmma_m)
    wiio, wiii = s[B].split(wii, wmma_n)

    s[B].reorder(ho, wo, hio, wio, hiio, wiio, hiii, wiii)

    s[B].tensorize(hiii, intrin_wmma_store_matrix(
        strides_dst=[n, 1], strides_from=[wn, 1], shape=(wmma_m, wmma_n, wmma_k), 
        out_dtype="float", A_shape=(wmma_m, wmma_n),
        C_shape=(wmma_m, wmma_n), C_scope="global"
    ))

    hwio = s[B].fuse(hio, wio)
    s[B].bind(ho, block_x)
    s[B].bind(wo, block_y)
    s[B].bind(hwio, thread_y)

    s[BF].compute_at(s[B], hwio)

    # Partition the reduction
    hbf, wbf = s[BF].op.axis
    hbfo, hbfi = s[BF].split(hbf, wmma_m)
    wbfo, wbfi = s[BF].split(wbf, wmma_n)

    (rbf, ) = s[BF].op.reduce_axis
    rbfo, rbfi = s[BF].split(rbf, wmma_k)

    s[BF].reorder(rbfo, hbfo, wbfo, hbfi, wbfi, rbfi)
    # Compute the 128B memory access length
    rbfoo, rbfoi = s[BF].split(rbfo, tk // wmma_k)

    A_ = te.placeholder((wmma_m, wmma_k), name="A", dtype=dtype)
    k_ = te.reduce_axis((0, wmma_k), name="k")

    if mode == "nn":
        B_= te.placeholder((wmma_k, wmma_n), name="B", dtype=dtype)
        C_ = te.compute(
            (wmma_m, wmma_n),
            lambda ii, jj: te.sum(A_[ii, k_].astype("float32") * B_[k_, jj].astype("float32"), axis=k_),
            name="C",
        )
        s[BF].tensorize(hbfi, intrin_wmma_gemm(
            AL_gemm=A_, WL_gemm=B_, CL_compute=C_, strides_A=[wk, 1], strides_W=[wn, 1], 
            strides_Conv=[wn, 1], shape=(wmma_m, wmma_n, wmma_k)
        ))
    else:
        B_ = te.placeholder((wmma_n, wmma_k), name="B", dtype=dtype)
        C_ = te.compute(
            (wmma_m, wmma_n),
            lambda ii, jj: te.sum(A_[ii, k_].astype("float32") * B_[jj, k_].astype("float32"), axis=k_),
            name="C",
        )
        s[BF].tensorize(hbfi, intrin_wmma_gemm(
            AL_gemm=A_, WL_gemm=B_, CL_compute=C_, strides_A=[wk, 1], strides_W=[wk, 1], 
            strides_Conv=[wn, 1], shape=(wmma_m, wmma_n, wmma_k)
        ))

    s[AF].compute_at(s[BF], rbfoi)
    s[WF].compute_at(s[BF], rbfoi)
    s[AS].compute_at(s[BF], rbfoo)
    s[WS].compute_at(s[BF], rbfoo)

    haf, waf = s[AF].op.axis
    hafo, hafi = s[AF].split(haf, wmma_m)
    s[AF].tensorize(hafi, intrin_wmma_load_matrix_A(
        strides_dst=[wk, 1], strides_from=[tk, 1], shape=(wmma_m, wmma_n, wmma_k), layout="row_major", 
        A_shape=(wmma_n, wmma_k), C_shape=(wmma_n, wmma_k), in_dtype=dtype, flags=flags, thread_extent=[thread_x, 32]
    ))

    hwf, wwf = s[WF].op.axis
    # Schedule shared -> Reg
    if mode == "nn": 
        wwfo, wwfi = s[WF].split(wwf, wmma_n)
        s[WF].reorder(wwfo, hwf, wwfi)
        s[WF].tensorize(hwf, intrin_wmma_load_matrix_W(
            strides_dst=[wn, 1], strides_from=[tn, 1], shape=(wmma_m, wmma_n, wmma_k), layout="row_major",
            A_shape=(wmma_k, wmma_n), C_shape=(wmma_k, wmma_n), in_dtype=dtype, flags=flags, thread_extent=[thread_x, 32]
        ))
    else:
        hwfo, hwfi = s[WF].split(hwf, wmma_n)
        s[WF].tensorize(hwfi, intrin_wmma_load_matrix_W(
            strides_dst=[wk, 1], strides_from=[tk, 1], shape=(wmma_m, wmma_n, wmma_k), layout="col_major",
            A_shape=(wmma_n, wmma_k), C_shape=(wmma_n, wmma_k), in_dtype=dtype, flags=flags, thread_extent=[thread_x, 32]
        ))

    # Schedule global -> shared
    if "xor" in flags:
        has, was = s[AS].op.axis
        if dtype == "float" or dtype == "float32":
            shape = (4, 32)
        else:
            shape = (4, 64)

        haso, hasi = s[AS].split(has, shape[0])
        waso, wasi = s[AS].split(was, shape[1])
        
        s[AS].reorder(haso, waso, hasi, wasi)
        hasoo, hasoi = s[AS].split(haso, num_warp)
        s[AS].bind(hasoi, thread_y)

        s[AS].tensorize(hasi, intrin_load_shared(
            strides_dst=[tk, 1], strides_from=[k, 1], shape=shape,
            in_dtype=dtype, layout="xor", thread_extent=[thread_x, 32]
        ))

        hws, wws = s[WS].op.axis
        hwso, hwsi = s[WS].split(hws, shape[0])
        wwso, wwsi = s[WS].split(wws, shape[1])

        s[WS].reorder(hwso, wwso, hwsi, wwsi)
        hwsoo, hwsoi = s[WS].split(hwso, num_warp)
        s[WS].bind(hwsoi, thread_y)

        if mode == "nn":
            if dtype == "float":
                s[WS].tensorize(hwsi, intrin_load_shared(
                    strides_dst=[tn, 1], strides_from=[n, 1], shape=shape,
                    in_dtype=dtype, layout="tf32_rhs_xor", thread_extent=[thread_x, 32]
                ))
            else:
                s[WS].tensorize(hwsi, intrin_load_shared(
                    strides_dst=[tn, 1], strides_from=[n, 1], shape=shape,
                    in_dtype=dtype, layout="xor", thread_extent=[thread_x, 32]
                ))
        else:
            s[WS].tensorize(hwsi, intrin_load_shared(
                strides_dst=[tk, 1], strides_from=[k, 1], shape=shape,
                in_dtype=dtype, layout="xor", thread_extent=[thread_x, 32]
            ))
    else:
        if dtype == "float" or dtype == "float32":
            vec_length = 4
        else:
            vec_length = 8

        has, was = s[AS].op.axis
        hwas = s[AS].fuse(has, was)
        hwaso, hwasi = s[AS].split(hwas, vec_length * 32)

        hwasoo, hwasoi = s[AS].split(hwaso, num_warp)
        hwasio, hwasii = s[AS].split(hwasi, vec_length)
        s[AS].bind(hwasoi, thread_y)
        s[AS].bind(hwasio, thread_x)
        s[AS].vectorize(hwasii)

        hws, wws = s[WS].op.axis
        hwws = s[WS].fuse(hws, wws)
        hwwso, hwwsi = s[WS].split(hwws, vec_length * 32)

        hwwsoo, hwwsoi = s[WS].split(hwwso, num_warp)
        hwwsio, hwwsii = s[WS].split(hwwsi, vec_length)
        s[WS].bind(hwwsoi, thread_y)
        s[WS].bind(hwwsio, thread_x)
        s[WS].vectorize(hwwsii)


    # print(tvm.lower(s, [A, W, B], simple_mode=True))
    dev = tvm.cuda(0)
    func = tvm.build(s, [A, W, B], "cuda")
    # print(func.imported_modules[0].get_source())
    

    a_np = np.random.uniform(size=(Ah, Aw)).astype(A.dtype)
    w_np = np.random.uniform(size=(Wh, Ww)).astype(W.dtype)

    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)

    b = tvm.nd.array(np.zeros((m, n), dtype=B.dtype), dev)

    func(a, w, b)
    """
        evaluator = func.time_evaluator(func.entry_name, dev, repeat=20, number=300)
        print(evaluator(a, w, b))
        print("Matmul: %f ms" % (evaluator(a, w, b).mean * 1e3))
    """
    if mode == "nn":
        tvm.testing.assert_allclose(b.numpy(), np.matmul(a.numpy(), w.numpy()), rtol = 0.001)
    else:
        tvm.testing.assert_allclose(b.numpy(), np.matmul(a.numpy(), np.transpose(w.numpy())), rtol = 0.001)


if __name__ == "__main__":
    test_tensor_core_gemm(
        shape=(8192, 512, 256), 
        tb_tile_shape=(128, 128, 32), 
        warp_tile_shape=(64, 64, 8), 
        wmma_shape=(16, 16, 8),
        dtype="float", mode="nn", flags=["ldmatrix", "xor"]
    )

    test_tensor_core_gemm(
        shape=(8192, 512, 256), 
        tb_tile_shape=(128, 128, 32), 
        warp_tile_shape=(64, 64, 8), 
        wmma_shape=(16, 16, 8),
        dtype="float", mode="nt", flags=["ldmatrix", "xor"]
    )
    
    test_tensor_core_gemm(
        shape=(8192, 512, 256), 
        tb_tile_shape=(256, 128, 64), 
        warp_tile_shape=(64, 64, 16), 
        wmma_shape=(16, 16, 16),
        dtype="float16", mode="nn", flags=["ldmatrix", "xor"]
    )

    test_tensor_core_gemm(
        shape=(8192, 512, 256), 
        tb_tile_shape=(256, 128, 64), 
        warp_tile_shape=(64, 64, 16), 
        wmma_shape=(16, 16, 16),
        dtype="float16", mode="nt", flags=["ldmatrix", "xor"]
    )
    
    test_tensor_core_gemm(
        shape=(8192, 512, 256), 
        tb_tile_shape=(128, 128, 32), 
        warp_tile_shape=(64, 64, 8), 
        wmma_shape=(16, 16, 8),
        dtype="float", mode="nn", flags=["ldmatrix"]
    )

    test_tensor_core_gemm(
        shape=(8192, 512, 256), 
        tb_tile_shape=(128, 128, 32), 
        warp_tile_shape=(64, 64, 8), 
        wmma_shape=(16, 16, 8),
        dtype="float", mode="nt", flags=["ldmatrix"]
    )
    
    test_tensor_core_gemm(
        shape=(8192, 512, 256), 
        tb_tile_shape=(256, 128, 64), 
        warp_tile_shape=(64, 64, 16), 
        wmma_shape=(16, 16, 16),
        dtype="float16", mode="nn", flags=["ldmatrix"]
    )

    test_tensor_core_gemm(
        shape=(8192, 512, 256), 
        tb_tile_shape=(256, 128, 64), 
        warp_tile_shape=(64, 64, 16), 
        wmma_shape=(16, 16, 16),
        dtype="float16", mode="nt", flags=["ldmatrix"]
    )
    test_tensor_core_gemm(
        shape=(8192, 512, 256), 
        tb_tile_shape=(128, 128, 32), 
        warp_tile_shape=(64, 64, 8), 
        wmma_shape=(16, 16, 8),
        dtype="float", mode="nn", flags=[]
    )

    test_tensor_core_gemm(
        shape=(8192, 512, 256), 
        tb_tile_shape=(128, 128, 32), 
        warp_tile_shape=(64, 64, 8), 
        wmma_shape=(16, 16, 8),
        dtype="float", mode="nt", flags=[]
    )
    
    test_tensor_core_gemm(
        shape=(8192, 512, 256), 
        tb_tile_shape=(256, 128, 64), 
        warp_tile_shape=(64, 64, 16), 
        wmma_shape=(16, 16, 16),
        dtype="float16", mode="nn", flags=[]
    )

    test_tensor_core_gemm(
        shape=(8192, 512, 256), 
        tb_tile_shape=(256, 128, 64), 
        warp_tile_shape=(64, 64, 16), 
        wmma_shape=(16, 16, 16),
        dtype="float16", mode="nt", flags=[]
    )