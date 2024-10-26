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

#[derive(Debug, Clone)]
pub struct Located<T> {
    pub t: T,
    pub line: usize,
    pub col: usize,
}

pub trait BoolToErrorHelper {
    fn or_err<E>(self, f: impl FnOnce() -> E) -> Result<(), E>;
}
impl BoolToErrorHelper for bool {
    #[inline(always)]
    fn or_err<E>(self, f: impl FnOnce() -> E) -> Result<(), E> {
        if !self {
            Err(f())
        } else {
            Ok(())
        }
    }
}
