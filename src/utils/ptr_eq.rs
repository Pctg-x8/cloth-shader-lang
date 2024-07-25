#[repr(transparent)]
pub struct PtrEq<'a, T: 'a + ?Sized>(pub &'a T);
impl<'a, T: 'a + ?Sized> Clone for PtrEq<'a, T> {
    fn clone(&self) -> Self {
        PtrEq(self.0)
    }
}
impl<'a, T: 'a + ?Sized> Copy for PtrEq<'a, T> {}
impl<'a, T: 'a + ?Sized> core::cmp::PartialEq for PtrEq<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.0, other.0)
    }
}
impl<'a, T: 'a + ?Sized> core::cmp::Eq for PtrEq<'a, T> {}
impl<'a, T: 'a + ?Sized> core::hash::Hash for PtrEq<'a, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.0 as *const T).hash(state)
    }
}
impl<'a, T: 'a + ?Sized> core::fmt::Debug for PtrEq<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ptr<{}>({:p})", core::any::type_name::<T>(), self.0)
    }
}
impl<'a, T: 'a + ?Sized> PtrEq<'a, T> {
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.0
    }
}
