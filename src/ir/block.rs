use std::{collections::HashMap, io::Write};

use typed_arena::Arena;

use crate::{
    concrete_type::{ConcreteType, IntrinsicType},
    parser::StatementNode,
    scope::{SymbolScope, VarId},
    source_ref::{SourceRef, SourceRefSliceEq},
    symbol::IntrinsicFunctionSymbol,
    utils::PtrEq,
};

use super::{
    expr::{binary_op, simplify_expression, simplify_lefthand_expression, ConstModifiers},
    ExprRef,
};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegisterRef(pub usize);
impl RegisterRef {
    pub fn ty<'c, 's>(self, ctx: &'c BlockGenerationContext<'_, 's>) -> &'c ConcreteType<'s> {
        ctx.register_type(self)
    }
}

#[derive(Debug, Clone)]
pub enum BlockFlowInstruction {
    Goto(usize),
    Conditional {
        source: RegisterRef,
        r#true: usize,
        r#false: usize,
    },
    Funcall {
        callee: RegisterRef,
        args: Vec<RegisterRef>,
        result: RegisterRef,
        after_return: Option<usize>,
    },
    StoreRef {
        ptr: RegisterRef,
        value: RegisterRef,
        after: Option<usize>,
    },
    Return(RegisterRef),
    Undetermined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicBinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    BitAnd,
    BitOr,
    BitXor,
    LeftShift,
    RightShift,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    LogAnd,
    LogOr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicUnaryOperation {
    Neg,
    BitNot,
    LogNot,
}

#[derive(Debug, Clone)]
pub enum BlockInstruction<'a, 's> {
    Cast {
        result: RegisterRef,
        src: RegisterRef,
        ty: ConcreteType<'s>,
    },
    InstantiateIntrinsicTypeClass {
        result: RegisterRef,
        src: RegisterRef,
        ty: IntrinsicType,
    },
    LoadRef {
        result: RegisterRef,
        ptr: RegisterRef,
    },
    ConstructIntrinsicComposite {
        result: RegisterRef,
        it: IntrinsicType,
        args: Vec<RegisterRef>,
    },
    ConstructTuple {
        result: RegisterRef,
        values: Vec<RegisterRef>,
    },
    ConstructStruct {
        result: RegisterRef,
        values: Vec<RegisterRef>,
    },
    PromoteIntToNumber {
        result: RegisterRef,
        value: RegisterRef,
    },
    IntrinsicBinaryOp {
        result: RegisterRef,
        left: RegisterRef,
        op: IntrinsicBinaryOperation,
        right: RegisterRef,
    },
    IntrinsicUnaryOp {
        result: RegisterRef,
        value: RegisterRef,
        op: IntrinsicUnaryOperation,
    },
    ConstUnit(RegisterRef),
    IntrinsicFunctionRef {
        result: RegisterRef,
        overloads: Vec<IntrinsicFunctionSymbol>,
    },
    IntrinsicTypeConstructorRef {
        result: RegisterRef,
        ty: IntrinsicType,
    },
    ScopeLocalVarRef {
        result: RegisterRef,
        scope: PtrEq<'a, SymbolScope<'a, 's>>,
        var_id: usize,
    },
    FunctionInputVarRef {
        result: RegisterRef,
        scope: PtrEq<'a, SymbolScope<'a, 's>>,
        var_id: usize,
    },
    UserDefinedFunctionRef {
        result: RegisterRef,
        scope: PtrEq<'a, SymbolScope<'a, 's>>,
        name: SourceRefSliceEq<'s>,
    },
    Swizzle {
        result: RegisterRef,
        source: RegisterRef,
        elements: Vec<usize>,
    },
    SwizzleRef {
        result: RegisterRef,
        source: RegisterRef,
        elements: Vec<usize>,
    },
    MemberRef {
        result: RegisterRef,
        source: RegisterRef,
        name: SourceRefSliceEq<'s>,
    },
    ArrayRef {
        result: RegisterRef,
        source: RegisterRef,
        index: RegisterRef,
    },
    TupleRef {
        result: RegisterRef,
        source: RegisterRef,
        index: usize,
    },
    ConstInt {
        result: RegisterRef,
        repr: SourceRefSliceEq<'s>,
        modifiers: ConstModifiers,
    },
    ConstNumber {
        result: RegisterRef,
        repr: SourceRefSliceEq<'s>,
        modifiers: ConstModifiers,
    },
    ConstUInt {
        result: RegisterRef,
        repr: SourceRefSliceEq<'s>,
        modifiers: ConstModifiers,
    },
    ConstSInt {
        result: RegisterRef,
        repr: SourceRefSliceEq<'s>,
        modifiers: ConstModifiers,
    },
    ConstFloat {
        result: RegisterRef,
        repr: SourceRefSliceEq<'s>,
        modifiers: ConstModifiers,
    },
    Phi {
        result: RegisterRef,
        incoming_selectors: HashMap<usize, RegisterRef>,
    },
}

#[derive(Debug, Clone)]
pub struct Block<'a, 's> {
    pub instructions: Vec<BlockInstruction<'a, 's>>,
    pub flow: BlockFlowInstruction,
}
impl<'a, 's> Block<'a, 's> {
    pub fn try_set_next(&mut self, next: usize) -> bool {
        match self.flow {
            BlockFlowInstruction::Undetermined => {
                self.flow = BlockFlowInstruction::Goto(next);
                true
            }
            BlockFlowInstruction::Funcall {
                ref mut after_return,
                ..
            } if after_return.is_none() => {
                *after_return = Some(next);
                true
            }
            BlockFlowInstruction::StoreRef { ref mut after, .. } if after.is_none() => {
                *after = Some(next);
                true
            }
            _ => false,
        }
    }

    pub fn try_set_branch(
        &mut self,
        condition: RegisterRef,
        r#true: usize,
        r#false: usize,
    ) -> bool {
        match self.flow {
            BlockFlowInstruction::Undetermined => {
                self.flow = BlockFlowInstruction::Conditional {
                    source: condition,
                    r#true,
                    r#false,
                };
                true
            }
            _ => false,
        }
    }
}

pub struct BlockInstructionEmitter<'c, 'a, 's> {
    pub generation_context: &'c mut BlockGenerationContext<'a, 's>,
    pub instructions: Vec<BlockInstruction<'a, 's>>,
}
impl<'c, 'a, 's> BlockInstructionEmitter<'c, 'a, 's> {
    #[inline]
    pub fn new(generation_context: &'c mut BlockGenerationContext<'a, 's>) -> Self {
        Self {
            generation_context,
            instructions: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn into_block(self, flow: BlockFlowInstruction) -> Block<'a, 's> {
        Block {
            instructions: self.instructions,
            flow,
        }
    }

    #[inline(always)]
    pub fn create_block(self, flow: BlockFlowInstruction) -> usize {
        let blk = Block {
            instructions: self.instructions,
            flow,
        };

        self.generation_context.add(blk)
    }

    pub fn into_instructions(self) -> Vec<BlockInstruction<'a, 's>> {
        self.instructions
    }

    #[inline]
    pub fn loaded(&mut self, ptr: RegisterRef) -> RegisterRef {
        match ptr.ty(self.generation_context) {
            ConcreteType::Ref(_) | ConcreteType::MutableRef(_) => self.load_ref(ptr),
            _ => ptr,
        }
    }

    pub fn cast(&mut self, src: RegisterRef, to: ConcreteType<'s>) -> RegisterRef {
        let dest_register = self.generation_context.alloc_register(to.clone());
        self.instructions.push(BlockInstruction::Cast {
            result: dest_register,
            src,
            ty: to,
        });

        dest_register
    }

    pub fn instantiate_intrinsic_type_class(
        &mut self,
        src: RegisterRef,
        to: IntrinsicType,
    ) -> RegisterRef {
        let dest_register = self.generation_context.alloc_register(to.into());
        self.instructions
            .push(BlockInstruction::InstantiateIntrinsicTypeClass {
                result: dest_register,
                src,
                ty: to,
            });

        dest_register
    }

    pub fn load_ref(&mut self, ptr: RegisterRef) -> RegisterRef {
        let dest_register = self.generation_context.alloc_register(
            self.generation_context
                .register_type(ptr)
                .as_dereferenced()
                .unwrap()
                .clone(),
        );
        self.instructions.push(BlockInstruction::LoadRef {
            result: dest_register,
            ptr,
        });

        dest_register
    }

    pub fn construct_intrinsic_composite(
        &mut self,
        it: IntrinsicType,
        args: Vec<RegisterRef>,
    ) -> RegisterRef {
        let dest_register = self.generation_context.alloc_register(it.into());
        self.instructions
            .push(BlockInstruction::ConstructIntrinsicComposite {
                result: dest_register,
                it,
                args,
            });

        dest_register
    }

    pub fn construct_tuple(&mut self, elements: Vec<RegisterRef>) -> RegisterRef {
        let ty = ConcreteType::Tuple(
            elements
                .iter()
                .map(|r| r.ty(self.generation_context).clone())
                .collect(),
        );

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions.push(BlockInstruction::ConstructTuple {
            result: dest_register,
            values: elements,
        });

        dest_register
    }

    pub fn construct_struct(
        &mut self,
        elements: Vec<RegisterRef>,
        out_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let dest_register = self.generation_context.alloc_register(out_type);
        self.instructions.push(BlockInstruction::ConstructStruct {
            result: dest_register,
            values: elements,
        });

        dest_register
    }

    pub fn promote_int_to_number(&mut self, r: RegisterRef) -> RegisterRef {
        let dest_register = self
            .generation_context
            .alloc_register(ConcreteType::UnknownNumberClass);
        self.instructions
            .push(BlockInstruction::PromoteIntToNumber {
                result: dest_register,
                value: r,
            });

        dest_register
    }

    pub fn intrinsic_binary_op(
        &mut self,
        left: RegisterRef,
        op: IntrinsicBinaryOperation,
        right: RegisterRef,
        out_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let dest_register = self.generation_context.alloc_register(out_type);
        self.instructions.push(BlockInstruction::IntrinsicBinaryOp {
            result: dest_register,
            left,
            op,
            right,
        });

        dest_register
    }

    pub fn intrinsic_unary_op(
        &mut self,
        value: RegisterRef,
        op: IntrinsicUnaryOperation,
        out_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let dest_register = self.generation_context.alloc_register(out_type);
        self.instructions.push(BlockInstruction::IntrinsicUnaryOp {
            result: dest_register,
            value,
            op,
        });

        dest_register
    }

    pub fn intrinsic_function_ref(
        &mut self,
        overloads: Vec<IntrinsicFunctionSymbol>,
    ) -> RegisterRef {
        let dest_register =
            self.generation_context
                .alloc_register(ConcreteType::OverloadedFunctions(
                    overloads
                        .iter()
                        .map(|s| (s.args.clone(), Box::new(s.output.clone())))
                        .collect(),
                ));
        self.instructions
            .push(BlockInstruction::IntrinsicFunctionRef {
                result: dest_register,
                overloads,
            });

        dest_register
    }

    pub fn intrinsic_type_constructor_ref(&mut self, it: IntrinsicType) -> RegisterRef {
        let dest_register = self
            .generation_context
            .alloc_register(ConcreteType::IntrinsicTypeConstructor(it));
        self.instructions
            .push(BlockInstruction::IntrinsicTypeConstructorRef {
                result: dest_register,
                ty: it,
            });

        dest_register
    }

    pub fn scope_local_var_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        var_id: usize,
    ) -> RegisterRef {
        let ty = scope.local_vars.borrow()[var_id].ty.clone().imm_ref();

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions.push(BlockInstruction::ScopeLocalVarRef {
            result: dest_register,
            scope: PtrEq(scope),
            var_id,
        });

        dest_register
    }

    pub fn scope_local_var_mutable_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        var_id: usize,
    ) -> RegisterRef {
        let ty = scope.local_vars.borrow()[var_id].ty.clone().mutable_ref();

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions.push(BlockInstruction::ScopeLocalVarRef {
            result: dest_register,
            scope: PtrEq(scope),
            var_id,
        });

        dest_register
    }

    pub fn function_input_var_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        var_id: usize,
    ) -> RegisterRef {
        let ty = scope.function_input_vars[var_id].ty.clone().imm_ref();

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions
            .push(BlockInstruction::FunctionInputVarRef {
                result: dest_register,
                scope: PtrEq(scope),
                var_id,
            });

        dest_register
    }

    pub fn function_input_var_mutable_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        var_id: usize,
    ) -> RegisterRef {
        let ty = scope.function_input_vars[var_id].ty.clone().mutable_ref();

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions
            .push(BlockInstruction::FunctionInputVarRef {
                result: dest_register,
                scope: PtrEq(scope),
                var_id,
            });

        dest_register
    }

    pub fn user_defined_function_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        name: SourceRef<'s>,
    ) -> RegisterRef {
        let fs = scope.user_defined_function_symbol(name.slice).unwrap();
        let ty = ConcreteType::Function {
            args: fs.inputs.iter().map(|(_, _, _, t)| t.clone()).collect(),
            output: match fs.output.len() {
                0 => None,
                1 => Some(Box::new(fs.output[0].1.clone())),
                _ => Some(Box::new(ConcreteType::Tuple(
                    fs.output.iter().map(|(_, t)| t.clone()).collect(),
                ))),
            },
        };

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions
            .push(BlockInstruction::UserDefinedFunctionRef {
                result: dest_register,
                scope: PtrEq(scope),
                name: SourceRefSliceEq(name),
            });

        dest_register
    }

    pub fn swizzle(&mut self, source: RegisterRef, elements: Vec<usize>) -> RegisterRef {
        let ty = ConcreteType::from(
            source
                .ty(self.generation_context)
                .scalar_type()
                .unwrap()
                .of_vector(elements.len().try_into().unwrap())
                .unwrap(),
        );

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions.push(BlockInstruction::Swizzle {
            result: dest_register,
            source,
            elements,
        });

        dest_register
    }

    pub fn swizzle_ref(&mut self, source: RegisterRef, elements: Vec<usize>) -> RegisterRef {
        let ty = ConcreteType::from(
            source
                .ty(self.generation_context)
                .scalar_type()
                .unwrap()
                .of_vector(elements.len().try_into().unwrap())
                .unwrap(),
        )
        .imm_ref();

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions.push(BlockInstruction::SwizzleRef {
            result: dest_register,
            source,
            elements,
        });

        dest_register
    }

    pub fn swizzle_mutable_ref(
        &mut self,
        source: RegisterRef,
        elements: Vec<usize>,
    ) -> RegisterRef {
        let ty = ConcreteType::from(
            source
                .ty(self.generation_context)
                .scalar_type()
                .unwrap()
                .of_vector(elements.len().try_into().unwrap())
                .unwrap(),
        )
        .mutable_ref();

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions.push(BlockInstruction::SwizzleRef {
            result: dest_register,
            source,
            elements,
        });

        dest_register
    }

    pub fn member_ref(
        &mut self,
        source: RegisterRef,
        name: SourceRef<'s>,
        member_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let ty = member_type.imm_ref();

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions.push(BlockInstruction::MemberRef {
            result: dest_register,
            source,
            name: SourceRefSliceEq(name),
        });

        dest_register
    }

    pub fn member_mutable_ref(
        &mut self,
        source: RegisterRef,
        name: SourceRef<'s>,
        member_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let ty = member_type.mutable_ref();

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions.push(BlockInstruction::MemberRef {
            result: dest_register,
            source,
            name: SourceRefSliceEq(name),
        });

        dest_register
    }

    pub fn array_ref(
        &mut self,
        source: RegisterRef,
        index: RegisterRef,
        element_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let ty = element_type.imm_ref();

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions.push(BlockInstruction::ArrayRef {
            result: dest_register,
            source,
            index,
        });

        dest_register
    }

    pub fn array_mutable_ref(
        &mut self,
        source: RegisterRef,
        index: RegisterRef,
        element_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let ty = element_type.mutable_ref();

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions.push(BlockInstruction::ArrayRef {
            result: dest_register,
            source,
            index,
        });

        dest_register
    }

    pub fn tuple_ref(
        &mut self,
        source: RegisterRef,
        index: usize,
        element_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let ty = element_type.imm_ref();

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions.push(BlockInstruction::TupleRef {
            result: dest_register,
            source,
            index,
        });

        dest_register
    }

    pub fn tuple_mutable_ref(
        &mut self,
        source: RegisterRef,
        index: usize,
        element_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let ty = element_type.mutable_ref();

        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions.push(BlockInstruction::TupleRef {
            result: dest_register,
            source,
            index,
        });

        dest_register
    }

    pub fn const_int(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        let dest_register = self
            .generation_context
            .alloc_register(ConcreteType::UnknownIntClass);
        self.instructions.push(BlockInstruction::ConstInt {
            result: dest_register,
            repr: SourceRefSliceEq(repr),
            modifiers: ConstModifiers::empty(),
        });

        dest_register
    }

    pub fn const_number(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        let dest_register = self
            .generation_context
            .alloc_register(ConcreteType::UnknownNumberClass);
        self.instructions.push(BlockInstruction::ConstNumber {
            result: dest_register,
            repr: SourceRefSliceEq(repr),
            modifiers: ConstModifiers::empty(),
        });

        dest_register
    }

    pub fn const_uint(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        let dest_register = self
            .generation_context
            .alloc_register(IntrinsicType::UInt.into());
        self.instructions.push(BlockInstruction::ConstUInt {
            result: dest_register,
            repr: SourceRefSliceEq(repr),
            modifiers: ConstModifiers::empty(),
        });

        dest_register
    }

    pub fn const_sint(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        let dest_register = self
            .generation_context
            .alloc_register(IntrinsicType::SInt.into());
        self.instructions.push(BlockInstruction::ConstSInt {
            result: dest_register,
            repr: SourceRefSliceEq(repr),
            modifiers: ConstModifiers::empty(),
        });

        dest_register
    }

    pub fn const_float(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        let dest_register = self
            .generation_context
            .alloc_register(IntrinsicType::Float.into());
        self.instructions.push(BlockInstruction::ConstFloat {
            result: dest_register,
            repr: SourceRefSliceEq(repr),
            modifiers: ConstModifiers::empty(),
        });

        dest_register
    }

    pub fn phi(
        &mut self,
        incoming_selectors: HashMap<usize, RegisterRef>,
        ty: ConcreteType<'s>,
    ) -> RegisterRef {
        let dest_register = self.generation_context.alloc_register(ty);
        self.instructions.push(BlockInstruction::Phi {
            result: dest_register,
            incoming_selectors,
        });

        dest_register
    }
}

pub struct BlockGenerationContext<'a, 's> {
    pub symbol_scope_arena: &'a Arena<SymbolScope<'a, 's>>,
    pub blocks: Vec<Block<'a, 's>>,
    pub registers: Vec<ConcreteType<'s>>,
}
impl<'a, 's> BlockGenerationContext<'a, 's> {
    #[inline]
    pub fn new(symbol_scope_arena: &'a Arena<SymbolScope<'a, 's>>) -> Self {
        Self {
            symbol_scope_arena,
            blocks: Vec::new(),
            registers: Vec::new(),
        }
    }

    pub fn create_block(
        &mut self,
        builder: impl FnOnce(&mut BlockInstructionEmitter),
        flow: BlockFlowInstruction,
    ) -> usize {
        let mut inst = BlockInstructionEmitter::new(self);
        builder(&mut inst);
        inst.create_block(flow)
    }

    pub fn alloc_register(&mut self, ty: ConcreteType<'s>) -> RegisterRef {
        self.registers.push(ty);

        RegisterRef(self.registers.len() - 1)
    }

    pub fn register_type(&self, r: RegisterRef) -> &ConcreteType<'s> {
        &self.registers[r.0]
    }

    pub fn add(&mut self, blk: Block<'a, 's>) -> usize {
        self.blocks.push(blk);

        self.blocks.len() - 1
    }

    #[inline(always)]
    pub fn block_mut(&mut self, index: usize) -> &mut Block<'a, 's> {
        &mut self.blocks[index]
    }

    #[inline(always)]
    pub fn try_chain(&mut self, from: usize, to: usize) -> bool {
        self.blocks[from].try_set_next(to)
    }

    pub fn dump_blocks(&self, writer: &mut (impl Write + ?Sized)) -> std::io::Result<()> {
        writeln!(writer, "Registers: ")?;
        for (n, r) in self.registers.iter().enumerate() {
            writeln!(writer, "  r{n}: {r:?}")?;
        }

        for (n, b) in self.blocks.iter().enumerate() {
            writeln!(writer, "b{n}: {{")?;

            if !b.instructions.is_empty() {
                for i in b.instructions.iter() {
                    match i {
                        BlockInstruction::Cast { result, src, ty } => {
                            writeln!(writer, "  r{} = r{} as {ty:?}", result.0, src.0)?
                        }
                        BlockInstruction::InstantiateIntrinsicTypeClass { result, src, ty } => {
                            writeln!(
                                writer,
                                "  r{} = Instantiate(r{} as {ty:?})",
                                result.0, src.0
                            )?
                        }
                        BlockInstruction::PromoteIntToNumber { result, value } => {
                            writeln!(writer, "  r{} = PromoteIntToNumber(r{})", result.0, value.0)?
                        }
                        BlockInstruction::ConstInt {
                            result,
                            repr,
                            modifiers,
                        } => writeln!(
                            writer,
                            "  r{} = ConstInt({repr:?}, mod={modifiers:?})",
                            result.0
                        )?,
                        BlockInstruction::ConstNumber {
                            result,
                            repr,
                            modifiers,
                        } => writeln!(
                            writer,
                            "  r{} = ConstNumber({repr:?}, mod={modifiers:?})",
                            result.0
                        )?,
                        BlockInstruction::ConstUInt {
                            result,
                            repr,
                            modifiers,
                        } => writeln!(
                            writer,
                            "  r{} = ConstUInt({repr:?}, mod={modifiers:?})",
                            result.0
                        )?,
                        BlockInstruction::ConstSInt {
                            result,
                            repr,
                            modifiers,
                        } => writeln!(
                            writer,
                            "  r{} = ConstSInt({repr:?}, mod={modifiers:?})",
                            result.0
                        )?,
                        BlockInstruction::ConstFloat {
                            result,
                            repr,
                            modifiers,
                        } => writeln!(
                            writer,
                            "  r{} = ConstFloat({repr:?}, mod={modifiers:?})",
                            result.0
                        )?,
                        BlockInstruction::ConstUnit(result) => {
                            writeln!(writer, "  r{} = ()", result.0)?
                        }
                        BlockInstruction::LoadRef { result, ptr } => {
                            writeln!(writer, "  r{} = Load r{}", result.0, ptr.0)?
                        }
                        BlockInstruction::ScopeLocalVarRef {
                            result,
                            scope,
                            var_id,
                        } => writeln!(
                            writer,
                            "  r{} = ScopeLocalVarRef({var_id}) in {scope:?}",
                            result.0
                        )?,
                        BlockInstruction::FunctionInputVarRef {
                            result,
                            scope,
                            var_id,
                        } => writeln!(
                            writer,
                            "  r{} = FunctionInputVarRef({var_id}) in {scope:?}",
                            result.0
                        )?,
                        BlockInstruction::MemberRef {
                            result,
                            source,
                            name,
                        } => writeln!(writer, "  r{} = ref r{}.({name:?})", result.0, source.0)?,
                        BlockInstruction::ArrayRef {
                            result,
                            source,
                            index,
                        } => writeln!(writer, "  r{} = ref r{}[r{}]", result.0, source.0, index.0)?,
                        BlockInstruction::TupleRef {
                            result,
                            source,
                            index,
                        } => writeln!(writer, "  r{} = ref r{}.{index}", result.0, source.0)?,
                        BlockInstruction::IntrinsicFunctionRef { result, overloads } => writeln!(
                            writer,
                            "  r{} = IntrinsicFunctionRef<{}>",
                            result.0,
                            overloads
                                .iter()
                                .map(|x| format!("{x:?}"))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )?,
                        BlockInstruction::IntrinsicTypeConstructorRef { result, ty } => writeln!(
                            writer,
                            "  r{} = IntrinsicTypeConstructorRef<{ty:?}>",
                            result.0
                        )?,
                        BlockInstruction::UserDefinedFunctionRef {
                            result,
                            scope,
                            name,
                        } => writeln!(
                            writer,
                            "  r{} = UserDefinedFunctionRef({name:?}) in {scope:?}",
                            result.0
                        )?,
                        BlockInstruction::SwizzleRef {
                            result,
                            source,
                            elements,
                        } => writeln!(
                            writer,
                            "  r{} = ref r{}.{}",
                            result.0,
                            source.0,
                            elements
                                .iter()
                                .map(|x| ['x', 'y', 'z', 'w'][*x])
                                .collect::<String>()
                        )?,
                        BlockInstruction::Swizzle {
                            result,
                            source,
                            elements,
                        } => writeln!(
                            writer,
                            "  r{} = r{}.{}",
                            result.0,
                            source.0,
                            elements
                                .iter()
                                .map(|x| ['x', 'y', 'z', 'w'][*x])
                                .collect::<String>()
                        )?,
                        BlockInstruction::Phi {
                            result,
                            incoming_selectors,
                        } => writeln!(
                            writer,
                            "  r{} = phi [{}]",
                            result.0,
                            incoming_selectors
                                .iter()
                                .map(|(from, r)| format!("{from} -> r{}", r.0))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )?,
                        BlockInstruction::ConstructIntrinsicComposite { result, it, args } => {
                            writeln!(
                                writer,
                                "  r{} = ConstructIntrinsicComposite<{it:?}>({})",
                                result.0,
                                args.iter()
                                    .map(|r| format!("r{}", r.0))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )?
                        }
                        BlockInstruction::ConstructTuple { result, values } => writeln!(
                            writer,
                            "  r{} = ({})",
                            result.0,
                            values
                                .iter()
                                .map(|r| format!("r{}", r.0))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )?,
                        BlockInstruction::ConstructStruct { result, values } => writeln!(
                            writer,
                            "  r{} = {{ {} }}",
                            result.0,
                            values
                                .iter()
                                .map(|r| format!("r{}", r.0))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )?,
                        BlockInstruction::IntrinsicBinaryOp {
                            result,
                            left,
                            op,
                            right,
                        } => writeln!(
                            writer,
                            "  r{} = {op:?}(r{}, r{})",
                            result.0, left.0, right.0
                        )?,
                        BlockInstruction::IntrinsicUnaryOp { result, value, op } => {
                            writeln!(writer, "  r{} = {op:?} r{}", result.0, value.0)?
                        }
                    }
                }

                writeln!(writer, "")?;
            }

            match b.flow {
                BlockFlowInstruction::Goto(x) => writeln!(writer, "  goto -> b{x}")?,
                BlockFlowInstruction::StoreRef {
                    ptr,
                    value,
                    after: Some(after),
                } => writeln!(writer, "  *r{} = r{} -> b{after}", ptr.0, value.0)?,
                BlockFlowInstruction::StoreRef {
                    ptr,
                    value,
                    after: None,
                } => writeln!(writer, "  *r{} = r{} -> ???", ptr.0, value.0)?,
                BlockFlowInstruction::Funcall {
                    callee,
                    ref args,
                    result,
                    after_return: Some(after_return),
                } => writeln!(
                    writer,
                    "  r{} = (r{})({}) -> b{after_return}",
                    result.0,
                    callee.0,
                    args.iter()
                        .map(|r| format!("r{}", r.0))
                        .collect::<Vec<_>>()
                        .join(", ")
                )?,
                BlockFlowInstruction::Funcall {
                    callee,
                    ref args,
                    result,
                    after_return: None,
                } => writeln!(
                    writer,
                    "  r{} = (r{})({}) -> ???",
                    result.0,
                    callee.0,
                    args.iter()
                        .map(|r| format!("r{}", r.0))
                        .collect::<Vec<_>>()
                        .join(", ")
                )?,
                BlockFlowInstruction::Conditional {
                    source,
                    r#true,
                    r#false,
                } => writeln!(
                    writer,
                    "  branch r{} ? -> b{} : -> b{}",
                    source.0, r#true, r#false
                )?,
                BlockFlowInstruction::Return(r) => writeln!(writer, "  return r{}", r.0)?,
                BlockFlowInstruction::Undetermined => writeln!(writer, "  -> ???")?,
            }

            writeln!(writer, "}}")?;
        }

        Ok(())
    }
}

pub fn transform_statement<'a, 's>(
    statement: StatementNode<'s>,
    scope: &'a SymbolScope<'a, 's>,
    ctx: &mut BlockGenerationContext<'a, 's>,
) -> (usize, usize) {
    match statement {
        StatementNode::Let {
            mut_token,
            varname_token,
            expr,
            ..
        } => {
            let expr_scope = ctx.symbol_scope_arena.alloc(scope.new_child());
            let expr = simplify_expression(expr, ctx, expr_scope);

            let mut inst = BlockInstructionEmitter::new(ctx);
            let expr_value = inst.loaded(expr.result);
            let VarId::ScopeLocal(var_id) = scope.declare_local_var(
                SourceRef::from(&varname_token),
                expr_value.ty(inst.generation_context).clone(),
                ExprRef(0), // TODO: 削除予定
                mut_token.is_some(),
            ) else {
                unreachable!();
            };
            let mutable_ref = inst.scope_local_var_mutable_ref(scope, var_id);
            let init_blk = inst.create_block(BlockFlowInstruction::StoreRef {
                ptr: mutable_ref,
                value: expr_value,
                after: None,
            });

            assert!(
                ctx.try_chain(expr.end_block, init_blk),
                "expr multiple out?"
            );

            (expr.start_block, init_blk)
        }
        StatementNode::OpEq {
            left_expr,
            opeq_token,
            expr,
        } => {
            let left_ref_expr = simplify_lefthand_expression(left_expr, ctx, scope);
            let right_expr_scope = ctx.symbol_scope_arena.alloc(scope.new_child());
            let right_expr = simplify_expression(expr, ctx, right_expr_scope);

            let mut inst = BlockInstructionEmitter::new(ctx);
            let right_value = inst.loaded(right_expr.result);
            let right_value = match opeq_token.slice {
                "=" => right_value,
                "+=" | "-=" | "*=" | "/=" | "%=" | "^^=" | "&=" | "|=" | "^=" | "<<=" | ">>="
                | "&&=" | "||=" => {
                    let left_expr_loaded = match left_ref_expr.result.ty(inst.generation_context) {
                        ConcreteType::Ref(_) | ConcreteType::MutableRef(_) => {
                            inst.load_ref(left_ref_expr.result)
                        }
                        _ => unreachable!("cannot store to right hand value"),
                    };

                    binary_op(
                        left_expr_loaded,
                        SourceRef {
                            slice: &opeq_token.slice[..opeq_token.slice.len() - 1],
                            line: opeq_token.line,
                            col: opeq_token.col,
                        },
                        right_expr.result,
                        &mut inst,
                    )
                }
                _ => unreachable!("unknown opeq token"),
            };
            let right_value = match left_ref_expr.result.ty(inst.generation_context) {
                ConcreteType::MutableRef(inner) => {
                    match (right_value.ty(inst.generation_context), &**inner) {
                        (a, b) if a == b => right_value,
                        (
                            ConcreteType::UnknownIntClass,
                            ConcreteType::Intrinsic(IntrinsicType::SInt),
                        ) => {
                            inst.instantiate_intrinsic_type_class(right_value, IntrinsicType::SInt)
                        }
                        (
                            ConcreteType::UnknownIntClass,
                            ConcreteType::Intrinsic(IntrinsicType::UInt),
                        ) => {
                            inst.instantiate_intrinsic_type_class(right_value, IntrinsicType::UInt)
                        }
                        (
                            ConcreteType::UnknownIntClass,
                            ConcreteType::Intrinsic(IntrinsicType::Float),
                        ) => {
                            inst.instantiate_intrinsic_type_class(right_value, IntrinsicType::Float)
                        }
                        (
                            ConcreteType::UnknownNumberClass,
                            ConcreteType::Intrinsic(IntrinsicType::Float),
                        ) => {
                            inst.instantiate_intrinsic_type_class(right_value, IntrinsicType::Float)
                        }
                        (res_ty, dest_ty) => {
                            panic!("Error: cannot assign: src={res_ty:?}, dest={dest_ty:?}")
                        }
                    }
                }
                ConcreteType::Ref(t) => {
                    panic!(
                        "Error: cannot assign to immutable variable: {:?}",
                        t.clone().imm_ref()
                    )
                }
                _ => panic!("Error: cannot assign to non-ref type value"),
            };

            let perform_block = inst.create_block(BlockFlowInstruction::StoreRef {
                ptr: left_ref_expr.result,
                value: right_value,
                after: None,
            });
            assert!(
                ctx.try_chain(left_ref_expr.end_block, right_expr.start_block),
                "left multiple out?"
            );
            assert!(
                ctx.try_chain(right_expr.end_block, perform_block),
                "right multiple out?"
            );

            (left_ref_expr.start_block, perform_block)
        }
        StatementNode::Expression(x) => {
            let expr = simplify_expression(x, ctx, scope);

            (expr.start_block, expr.end_block)
        }
        StatementNode::While { .. } => unimplemented!("while"),
    }
}
