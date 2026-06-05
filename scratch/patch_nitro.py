import re

with open("kernel_nitro.py", "r") as f:
    content = f.read()

new_backward = """
    @staticmethod
    def backward(ctx, dout, dstate, dn_out):
        q_s, k_s, v_s, a_s, b_s, is_s, in_s = ctx.saved_tensors
        B, H, T = q_s.shape[:3]
        dk = q_s.shape[3]
        dv = v_s.shape[3]
        chunk_size = ctx.chunk_size
        has_init = ctx.has_init_s
        has_init_n = ctx.has_init_n

        num_chunks = (T + chunk_size - 1) // chunk_size
        states = torch.empty((B, H, num_chunks, dk, dv), device=q_s.device, dtype=torch.float32)
        out_dummy = torch.empty((B, H, T, dv), device=q_s.device, dtype=torch.float32)
        n_dummy   = torch.empty((B, H, T, dk), device=q_s.device, dtype=torch.float32)
        
        grid_fwd = (B, H)
        _chunk_fwd_kernel_fused_sn[grid_fwd](
            q_s, k_s, v_s, a_s, b_s, out_dummy, states, n_dummy, is_s, in_s, T,
            has_init, has_init_n,
            dk, dv, chunk_size,
            q_s.stride(0), q_s.stride(1), q_s.stride(2), q_s.stride(3),
            k_s.stride(0), k_s.stride(1), k_s.stride(2), k_s.stride(3),
            v_s.stride(0), v_s.stride(1), v_s.stride(2), v_s.stride(3),
            a_s.stride(0), a_s.stride(1), a_s.stride(2),
            b_s.stride(0), b_s.stride(1), b_s.stride(2),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            n_dummy.stride(0), n_dummy.stride(1), n_dummy.stride(2), n_dummy.stride(3),
            is_s.stride(0) if has_init else 0,
            is_s.stride(1) if has_init else 0,
            is_s.stride(2) if has_init else 0,
            is_s.stride(3) if has_init else 0,
            in_s.stride(0) if has_init_n else 0,
            in_s.stride(1) if has_init_n else 0,
            in_s.stride(2) if has_init_n else 0,
            num_warps=4, num_stages=2,
        )

        dout_s = dout.transpose(1, 2).contiguous().float()
        dq = torch.empty_like(q_s)
        dk_ = torch.empty_like(k_s)
        dv_ = torch.empty_like(v_s)
        da = torch.empty_like(a_s)
        db = torch.empty_like(b_s)

        if dstate is not None:
            dstate_s = dstate.float().contiguous()
            ds_strides = dstate_s.stride()
        else:
            dstate_s = torch.empty(0, device=dout.device, dtype=torch.float32)
            ds_strides = (0, 0, 0, 0)

        if has_init:
            dinitial_state = torch.empty((B, H, dk, dv), device=dout.device, dtype=torch.float32).contiguous()
            dis_strides = dinitial_state.stride()
        else:
            dinitial_state = torch.empty(0, device=dout.device, dtype=torch.float32)
            dis_strides = (0, 0, 0, 0)

        grid = (B, H)
        _chunk_bwd_kernel[grid](
            q_s, k_s, v_s, a_s, b_s, states, dout_s,
            dq, dk_, dv_, da, db,
            dstate_s, dinitial_state,
            is_s,
            T,
            dk, dv, chunk_size,
            dstate is not None, has_init,
            has_init,
            q_s.stride(0), q_s.stride(1), q_s.stride(2), q_s.stride(3),
            k_s.stride(0), k_s.stride(1), k_s.stride(2), k_s.stride(3),
            v_s.stride(0), v_s.stride(1), v_s.stride(2), v_s.stride(3),
            a_s.stride(0), a_s.stride(1), a_s.stride(2),
            b_s.stride(0), b_s.stride(1), b_s.stride(2),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            ds_strides[0], ds_strides[1], ds_strides[2], ds_strides[3],
            dis_strides[0], dis_strides[1], dis_strides[2], dis_strides[3],
            is_s.stride(0) if has_init else 0,
            is_s.stride(1) if has_init else 0,
            is_s.stride(2) if has_init else 0,
            is_s.stride(3) if has_init else 0,
            num_warps=4,
            num_stages=2,
        )

        if dn_out is not None:
            dn_out_s = dn_out.transpose(1, 2).contiguous().float()
            dk_n = torch.empty_like(k_s)
            da_n = torch.empty_like(a_s)
            db_n = torch.empty_like(b_s)
            
            if has_init_n:
                dinitial_n = torch.empty((B, H, dk), device=dout.device, dtype=torch.float32).contiguous()
                din_strides = dinitial_n.stride()
            else:
                dinitial_n = torch.empty(0, device=dout.device, dtype=torch.float32)
                din_strides = (0, 0, 0)

            _vec_recurrence_bwd_kernel[grid](
                a_s, b_s, k_s, n_dummy, dn_out_s,
                da_n, db_n, dk_n,
                dinitial_n,
                in_s,
                T,
                has_init_n, dk, chunk_size,
                a_s.stride(0), a_s.stride(1), a_s.stride(2),
                b_s.stride(0), b_s.stride(1), b_s.stride(2),
                k_s.stride(0), k_s.stride(1), k_s.stride(2), k_s.stride(3),
                n_dummy.stride(0), n_dummy.stride(1), n_dummy.stride(2), n_dummy.stride(3),
                da_n.stride(0), da_n.stride(1), da_n.stride(2),
                db_n.stride(0), db_n.stride(1), db_n.stride(2),
                dk_n.stride(0), dk_n.stride(1), dk_n.stride(2), dk_n.stride(3),
                in_s.stride(0) if has_init_n else 0,
                in_s.stride(1) if has_init_n else 0,
                in_s.stride(2) if has_init_n else 0,
                num_warps=4, num_stages=2,
            )
            dk_ += dk_n
            da += da_n
            db += db_n
        else:
            dinitial_n = None

        dtype = ctx.dtype
        return (
            dq.transpose(1,2).to(dtype),
            dk_.transpose(1,2).to(dtype),
            dv_.transpose(1,2).to(dtype),
            da.transpose(1,2).to(dtype),
            db.transpose(1,2).to(dtype),
            dinitial_state.to(dtype) if has_init else None,
            dinitial_n.to(dtype) if has_init_n else None,
            None,
        )
"""

backward_pattern = re.compile(
    r"    @staticmethod\n    def backward\(ctx, dout, dstate, dn_out\):.*?        \)\n",
    re.DOTALL
)

forward_pattern = re.compile(
    r"        ctx\.save_for_backward\(q_s, k_s, v_s, a_s, b_s, is_s\)\n        ctx\.dtype      = dtype\n        ctx\.chunk_size = chunk_size\n        ctx\.has_init_s = has_init_s"
)
forward_replacement = """        ctx.save_for_backward(q_s, k_s, v_s, a_s, b_s, is_s, in_s)
        ctx.dtype      = dtype
        ctx.chunk_size = chunk_size
        ctx.has_init_s = has_init_s
        ctx.has_init_n = has_init_n"""

content = backward_pattern.sub(new_backward, content)
content = forward_pattern.sub(forward_replacement, content)

with open("kernel_nitro.py", "w") as f:
    f.write(content)

print("Patched kernel_nitro.py!")
