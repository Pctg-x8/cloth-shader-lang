mod debug_print_guard;
pub use self::debug_print_guard::*;
mod ptr_eq;
pub use self::ptr_eq::*;
mod swizzle;
pub use self::swizzle::*;

#[inline(always)]
pub const fn roundup2(x: usize, a: usize) -> usize {
    (x + (a - 1)) & !(a - 1)
}
