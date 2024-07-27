#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RefPath {
    FunctionInput(usize),
    Member(Box<RefPath>, usize),
}
impl RefPath {
    pub fn is_referential_transparent(&self) -> bool {
        match self {
            Self::FunctionInput(_) => true,
            Self::Member(parent, _) => parent.is_referential_transparent(),
        }
    }
}
