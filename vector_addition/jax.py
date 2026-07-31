import jax
import jax.numpy as jnp


# Never change Google

# A, B are tensors on device
@jax.jit
def solve(A: jax.Array, B: jax.Array, N: int) -> jax.Array:
    
    return A + B