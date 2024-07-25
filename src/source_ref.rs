use crate::parser::Token;

#[derive(Debug, Clone)]
pub struct SourceRef<'s> {
    pub slice: &'s str,
    pub line: usize,
    pub col: usize,
}
impl<'s> From<&'_ Token<'s>> for SourceRef<'s> {
    #[inline]
    fn from(value: &'_ Token<'s>) -> Self {
        Self {
            slice: value.slice,
            line: value.line,
            col: value.col,
        }
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct SourceRefSliceEq<'s>(pub SourceRef<'s>);
impl<'s> From<&'_ Token<'s>> for SourceRefSliceEq<'s> {
    #[inline]
    fn from(value: &'_ Token<'s>) -> Self {
        Self(SourceRef::from(value))
    }
}
impl core::fmt::Debug for SourceRefSliceEq<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <SourceRef as core::fmt::Debug>::fmt(&self.0, f)
    }
}
impl core::cmp::PartialEq for SourceRefSliceEq<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.0.slice == other.0.slice
    }
}
impl core::cmp::Eq for SourceRefSliceEq<'_> {}
impl core::hash::Hash for SourceRefSliceEq<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.slice.hash(state)
    }
}
