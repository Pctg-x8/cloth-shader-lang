#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RefPath<'s> {
    FunctionInput(usize),
    Member(Box<RefPath<'s>>, &'s str),
}
