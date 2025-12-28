import torch
import triton
import triton.language as tl
import threading

@triton.jit
def _get_thread_id() -> tl.int32:
    return tl.inline_asm_elementwise(
        "mov.u32 $0, %tid.x;",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )

# 测试 1: 多 warp，不带 bar_sync
@triton.jit  
def _barrier_no_sync(barrier_ptr, num_programs: tl.constexpr, phase: tl.constexpr):
    counter_ptr = barrier_ptr + phase
    tid = _get_thread_id()
    
    if tid == 0:
        arrived = tl.atomic_add(counter_ptr, 1, sem="acq_rel")
        if arrived == num_programs - 1:
            tl.atomic_xchg(counter_ptr, 0, sem="release")
        else:
            while tl.atomic_cas(counter_ptr, 0, 0) != 0:
                pass


@triton.jit
def _test_multi_warp_no_sync(output_ptr, barrier_ptr, num_programs: tl.constexpr):
    pid = tl.program_id(0)
    tid = _get_thread_id()
    
    if tid == 0:
        tl.store(output_ptr + pid, 1)
    
    _barrier_no_sync(barrier_ptr, num_programs, phase=0)
    
    if tid == 0:
        tl.store(output_ptr + num_programs + pid, 2)
    
    _barrier_no_sync(barrier_ptr, num_programs, phase=1)
    
    if tid == 0:
        tl.store(output_ptr + num_programs * 2 + pid, 3)


# 测试 2: 多 warp，带 tl.debug_barrier
@triton.jit  
def _barrier_with_debug(barrier_ptr, num_programs: tl.constexpr, phase: tl.constexpr):
    counter_ptr = barrier_ptr + phase
    tid = _get_thread_id()
    
    if tid == 0:
        arrived = tl.atomic_add(counter_ptr, 1, sem="acq_rel")
        if arrived == num_programs - 1:
            tl.atomic_xchg(counter_ptr, 0, sem="release")
        else:
            while tl.atomic_cas(counter_ptr, 0, 0) != 0:
                pass
    
    tl.debug_barrier()


@triton.jit
def _test_multi_warp_with_debug(output_ptr, barrier_ptr, num_programs: tl.constexpr):
    pid = tl.program_id(0)
    tid = _get_thread_id()
    
    if tid == 0:
        tl.store(output_ptr + pid, 1)
    
    _barrier_with_debug(barrier_ptr, num_programs, phase=0)
    
    if tid == 0:
        tl.store(output_ptr + num_programs + pid, 2)
    
    _barrier_with_debug(barrier_ptr, num_programs, phase=1)
    
    if tid == 0:
        tl.store(output_ptr + num_programs * 2 + pid, 3)


def test_version(name, kernel_fn, num_warps):
    num_programs = 4
    output = torch.zeros(num_programs * 3, device="cuda", dtype=torch.int32)
    barrier = torch.zeros(2, device="cuda", dtype=torch.int32)
    
    def run():
        kernel_fn[(num_programs,)](
            output_ptr=output,
            barrier_ptr=barrier,
            num_programs=num_programs,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()
    
    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=3)
    
    if thread.is_alive():
        print(f"{name} (num_warps={num_warps}): TIMEOUT!")
    else:
        print(f"{name} (num_warps={num_warps}): SUCCESS! output={output.tolist()}")


if __name__ == "__main__":
    print("=== 不带 bar_sync 的 barrier ===")
    for nw in [1, 2, 4, 8]:
        test_version("no_sync", _test_multi_warp_no_sync, nw)
        torch.cuda.empty_cache()
    
    print("\n=== 带 tl.debug_barrier 的 barrier ===")
    for nw in [1, 2, 4, 8]:
        test_version("with_debug", _test_multi_warp_with_debug, nw)
        torch.cuda.empty_cache()