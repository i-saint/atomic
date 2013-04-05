public atomic_add8
public atomic_add16
public atomic_add32
public atomic_add64

public atomic_swap8
public atomic_swap16
public atomic_swap32
public atomic_swap64

public atomic_cas8
public atomic_cas16
public atomic_cas32
public atomic_cas64

.code

atomic_add8:
    lock xadd byte ptr [rcx], dl
    mov al, dl
    ret

atomic_add16:
    lock xadd word ptr [rcx], dx
    mov ax, dx
    ret

atomic_add32:
    lock xadd dword ptr [rcx], edx
    mov eax, edx
    ret

atomic_add64:
    lock xadd qword ptr [rcx], rdx
    mov rax, rdx
    ret


atomic_swap8:
    lock xchg byte ptr [rcx], dl
    mov al, dl
    ret

atomic_swap16:
    lock xchg word ptr [rcx], dx
    mov ax, dx
    ret

atomic_swap32:
    lock xchg dword ptr [rcx], edx
    mov eax, edx
    ret

atomic_swap64:
    lock xchg qword ptr [rcx], rdx
    mov rax, rdx
    ret


atomic_cas8:
    mov al, dl
    lock cmpxchg byte ptr [rcx], r8b
    ret

atomic_cas16:
    mov ax, dx
    lock cmpxchg word ptr [rcx], r8w
    ret

atomic_cas32:
    mov eax, edx
    lock cmpxchg dword ptr [rcx], r8d
    ret

atomic_cas64:
    mov rax, rdx
    lock cmpxchg qword ptr [rcx], r8
    ret

end
