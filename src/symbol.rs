use meta::SymbolAttribute;

use crate::{
    concrete_type::{ConcreteType, IntrinsicType, RefStorageClass, UserDefinedType},
    ir::ExprRef,
    scope::SymbolScope,
    source_ref::SourceRef,
};

pub mod meta;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntrinsicFunctionSymbol {
    pub name: &'static str,
    pub args: Vec<ConcreteType<'static>>,
    pub output: ConcreteType<'static>,
    pub is_pure: bool,
}

#[derive(Debug, Clone)]
pub struct FunctionInputVariable<'s> {
    pub occurence: SourceRef<'s>,
    pub ty: ConcreteType<'s>,
    pub mutable: bool,
    pub storage_class: RefStorageClass,
}

#[derive(Debug, Clone)]
pub struct LocalVariable<'s> {
    pub occurence: SourceRef<'s>,
    pub ty: ConcreteType<'s>,
    pub mutable: bool,
}

#[derive(Debug, Clone)]
pub struct UserDefinedFunctionInput<'s> {
    pub attributes: SymbolAttribute,
    pub mutable: bool,
    pub name: SourceRef<'s>,
    pub ty: ConcreteType<'s>,
}

#[derive(Debug, Clone)]
pub struct UserDefinedFunctionSymbol<'s> {
    pub occurence: SourceRef<'s>,
    pub attribute: SymbolAttribute,
    pub inputs: Vec<UserDefinedFunctionInput<'s>>,
    pub output: Vec<(SymbolAttribute, ConcreteType<'s>)>,
}
impl<'s> UserDefinedFunctionSymbol<'s> {
    #[inline(always)]
    pub const fn name(&self) -> &'s str {
        self.occurence.slice
    }

    #[inline(always)]
    pub fn concrete_type(&self) -> ConcreteType<'s> {
        ConcreteType::Function {
            args: self.inputs.iter().map(|x| x.ty.clone()).collect(),
            output: match &self.output[..] {
                &[] => None,
                &[(_, ref t)] => Some(Box::new(t.clone())),
                xs => Some(Box::new(ConcreteType::Tuple(
                    xs.iter().map(|(_, t)| t.clone()).collect(),
                ))),
            },
        }
    }

    pub fn flatten_output<'a>(
        &self,
        function_scope: &'a SymbolScope<'a, 's>,
    ) -> Vec<(SymbolAttribute, IntrinsicType)> {
        fn rec<'a, 's>(
            sink: &mut Vec<(SymbolAttribute, IntrinsicType)>,
            symbol_attribute: &SymbolAttribute,
            member_ty: &ConcreteType<'s>,
            function_scope: &'a SymbolScope<'a, 's>,
        ) {
            match member_ty {
                &ConcreteType::Intrinsic(it) => {
                    // can output directly
                    sink.push((symbol_attribute.clone(), it));
                }
                ConcreteType::Struct(members) => {
                    // flatten more
                    for x in members {
                        rec(sink, &x.attribute, &x.ty, function_scope);
                    }
                }
                ConcreteType::Tuple(xs) => {
                    // flatten more
                    let default_attribute = SymbolAttribute::default();

                    for x in xs {
                        rec(sink, &default_attribute, x, function_scope);
                    }
                }
                ConcreteType::UserDefined { name, .. } => {
                    match function_scope.lookup_user_defined_type(name) {
                        Some((_, (_, UserDefinedType::Struct(members)))) => {
                            // flatten more
                            for x in members {
                                rec(sink, &x.attribute, &x.ty, function_scope);
                            }
                        }
                        None => panic!("Error: cannot output opaque type"),
                    }
                }
                _ => panic!("Error: cannot output this type: {member_ty:?}"),
            }
        }

        let mut flattened = Vec::new();
        for (a, c) in &self.output {
            rec(&mut flattened, a, c, function_scope);
        }
        flattened
    }
}
