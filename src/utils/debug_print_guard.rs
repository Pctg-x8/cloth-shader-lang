#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DebugPrintGuard<T>(pub T);
impl<T> core::fmt::Debug for DebugPrintGuard<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(masked {}...)", core::any::type_name::<T>())
    }
}
