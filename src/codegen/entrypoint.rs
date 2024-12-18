use crate::{
    concrete_type::{ConcreteType, IntrinsicScalarType, IntrinsicType},
    ref_path::RefPath,
    scope::SymbolScope,
    spirv as spv,
    symbol::{
        meta::{BuiltinInputOutput, ShaderModel, SymbolAttribute},
        UserDefinedFunctionSymbol,
    },
    utils::roundup2,
};

#[derive(Debug)]
pub struct ShaderInterfaceInputVariable {
    pub ty: spv::Type,
    pub original_refpath: RefPath,
    pub decorations: Vec<spv::Decorate>,
}

#[derive(Debug)]
pub struct ShaderInterfaceOutputVariable {
    pub ty: spv::Type,
    pub decorations: Vec<spv::Decorate>,
}

#[derive(Debug)]
pub struct ShaderInterfaceUniformVariable {
    pub ty: spv::Type,
    pub original_refpath: RefPath,
    pub decorations: Vec<spv::Decorate>,
}

#[derive(Debug)]
pub struct ShaderInterfacePushConstantVariable {
    pub ty: spv::Type,
    pub original_refpath: RefPath,
    pub offset: u32,
}

#[derive(Debug)]
pub struct WorkgroupSharedVariable {
    pub ty: spv::Type,
    pub original_refpath: RefPath,
}

#[derive(Debug)]
pub struct ShaderEntryPointGlobalVariables {
    pub inputs: Vec<ShaderInterfaceInputVariable>,
    pub outputs: Vec<ShaderInterfaceOutputVariable>,
    pub uniforms: Vec<ShaderInterfaceUniformVariable>,
    pub push_constants: Vec<ShaderInterfacePushConstantVariable>,
    pub workgroup_shared_vars: Vec<WorkgroupSharedVariable>,
}

#[derive(Debug)]
pub struct ShaderEntryPointDescription<'s> {
    pub name: &'s str,
    pub execution_model: spv::asm::ExecutionModel,
    pub execution_mode_modifiers: Vec<spv::ExecutionModeModifier>,
    pub global_variables: ShaderEntryPointGlobalVariables,
}
impl<'s> ShaderEntryPointDescription<'s> {
    pub fn extract(func: &UserDefinedFunctionSymbol<'s>, scope: &SymbolScope<'_, 's>) -> Self {
        let execution_model = match func.attribute.shader_model {
            Some(ShaderModel::VertexShader) => spv::asm::ExecutionModel::Vertex,
            Some(ShaderModel::TessellationControlShader) => {
                spv::asm::ExecutionModel::TessellationControl
            }
            Some(ShaderModel::TessellationEvaluationShader) => {
                spv::asm::ExecutionModel::TessellationEvaluation
            }
            Some(ShaderModel::GeometryShader) => spv::asm::ExecutionModel::Geometry,
            Some(ShaderModel::FragmentShader) => spv::asm::ExecutionModel::Fragment,
            Some(ShaderModel::ComputeShader) => spv::asm::ExecutionModel::GLCompute,
            None => unreachable!("not a entry point function"),
        };
        let execution_mode_modifiers = if execution_model == spv::asm::ExecutionModel::Fragment {
            vec![spv::ExecutionModeModifier::OriginUpperLeft]
        } else {
            vec![]
        };

        let mut global_vars = ShaderEntryPointGlobalVariables {
            inputs: Vec::new(),
            outputs: Vec::new(),
            uniforms: Vec::new(),
            push_constants: Vec::new(),
            workgroup_shared_vars: Vec::new(),
        };
        for (n, (attr, _, _, ty)) in func.inputs.iter().enumerate() {
            process_entry_point_inputs(
                attr,
                &RefPath::FunctionInput(n),
                ty,
                scope,
                &mut global_vars,
            );
        }
        if func.output.len() > 1 || func.output[0].1 != IntrinsicType::Scalar(IntrinsicScalarType::Unit).into() {
            for (attr, ty) in func.output.iter() {
                process_entry_point_outputs(attr, ty, scope, &mut global_vars.outputs);
            }
        }

        Self {
            name: func.occurence.slice,
            execution_model,
            execution_mode_modifiers,
            global_variables: global_vars,
        }
    }
}

fn process_entry_point_inputs<'s>(
    attr: &SymbolAttribute,
    refpath: &RefPath,
    ty: &ConcreteType<'s>,
    scope: &SymbolScope<'_, 's>,
    global_vars: &mut ShaderEntryPointGlobalVariables,
) {
    match attr {
        SymbolAttribute {
            module_entry_point: true,
            ..
        } => panic!("Error: entry point input cannot be a module entry point"),
        SymbolAttribute {
            shader_model: Some(_),
            ..
        } => panic!("Error: entry point input cannot have any shader model attributes"),
        SymbolAttribute {
            descriptor_set_location: Some(set),
            descriptor_set_binding: Some(binding),
            image_format_specifier,
            input_attachment_index: None,
            push_constant_offset: None,
            bound_location: None,
            bound_builtin_io: None,
            workgroup_shared: false,
            ..
        } => match ty {
            ConcreteType::Struct(members) => {
                let spv_ty = spv::Type::Struct {
                    decorations: vec![spv::Decorate::Block],
                    member_types: members
                        .iter()
                        .scan(0, |top, x| {
                            if !x.attribute.is_empty() {
                                panic!("a member of an uniform struct cannot have any attributes");
                            }

                            let offset = roundup2(
                                *top,
                                x.ty.std140_alignment()
                                    .expect("this type cannot be a member of an uniform struct"),
                            );
                            *top = offset
                                + x.ty
                                    .std140_size()
                                    .expect("this type cannot be a member of an uniform struct");

                            let ty = x.ty.make_spv_type(scope);
                            let mut decorations = vec![spv::Decorate::Offset(offset as _)];
                            if let spv::Type::Matrix(ref mt, _) = ty {
                                decorations.extend([
                                    spv::Decorate::ColMajor,
                                    spv::Decorate::MatrixStride(mt.matrix_stride().unwrap()),
                                ]);
                            }

                            Some(spv::TypeStructMember { ty, decorations })
                        })
                        .collect(),
                };

                global_vars.uniforms.push(ShaderInterfaceUniformVariable {
                    ty: spv_ty,
                    original_refpath: refpath.clone(),
                    decorations: vec![
                        spv::Decorate::DescriptorSet(*set),
                        spv::Decorate::Binding(*binding),
                    ],
                });
            }
            ConcreteType::Intrinsic(IntrinsicType::Texture2D) => {
                let spv_ty = spv::Type::SampledImage {
                    image_type: Box::new(spv::Type::Image {
                        sampled_type: spv::ScalarType::Float(32).into(),
                        dim: spv::asm::Dim::Dim2,
                        depth: Some(false),
                        arrayed: false,
                        multisampled: false,
                        sampled: spv::asm::TypeImageSampled::WithSamplingOps,
                        image_format: spv::asm::ImageFormat::Rgba8,
                        access_qualifier: None,
                    }),
                };

                global_vars.uniforms.push(ShaderInterfaceUniformVariable {
                    ty: spv_ty,
                    original_refpath: refpath.clone(),
                    decorations: vec![
                        spv::Decorate::DescriptorSet(*set),
                        spv::Decorate::Binding(*binding),
                    ],
                })
            }
            ConcreteType::Intrinsic(IntrinsicType::Image2D) => {
                let spv_ty = spv::Type::Image {
                    sampled_type: spv::ScalarType::Float(32).into(),
                    dim: spv::asm::Dim::Dim2,
                    depth: Some(false),
                    arrayed: false,
                    multisampled: false,
                    sampled: spv::asm::TypeImageSampled::WithSamplingOps,
                    image_format: image_format_specifier.unwrap_or(spv::asm::ImageFormat::Rgba8),
                    access_qualifier: None,
                };

                global_vars.uniforms.push(ShaderInterfaceUniformVariable {
                    ty: spv_ty,
                    original_refpath: refpath.clone(),
                    decorations: vec![
                        spv::Decorate::DescriptorSet(*set),
                        spv::Decorate::Binding(*binding),
                    ],
                })
            }
            ty => {
                let ty = spv::Type::Struct {
                    member_types: vec![spv::TypeStructMember {
                        ty: ty.make_spv_type(scope),
                        decorations: vec![spv::Decorate::Offset(0)],
                    }],
                    decorations: vec![spv::Decorate::Block],
                };

                global_vars.uniforms.push(ShaderInterfaceUniformVariable {
                    ty,
                    original_refpath: refpath.clone(),
                    decorations: vec![
                        spv::Decorate::DescriptorSet(*set),
                        spv::Decorate::Binding(*binding),
                    ],
                })
            }
        },
        SymbolAttribute {
            descriptor_set_location: Some(set),
            descriptor_set_binding: Some(binding),
            input_attachment_index: Some(aix),
            push_constant_offset: None,
            bound_location: None,
            bound_builtin_io: None,
            workgroup_shared: false,
            ..
        } => match ty {
            ConcreteType::Intrinsic(IntrinsicType::SubpassInput) => {
                global_vars.uniforms.push(ShaderInterfaceUniformVariable {
                    ty: spv::Type::SUBPASS_DATA_IMAGE_TYPE,
                    original_refpath: refpath.clone(),
                    decorations: vec![
                        spv::Decorate::DescriptorSet(*set),
                        spv::Decorate::Binding(*binding),
                        spv::Decorate::InputAttachmentIndex(*aix),
                    ],
                })
            }
            _ => panic!(
                "Error: descriptor subpass input binding canbe done only for the SubpassInput type"
            ),
        },
        SymbolAttribute {
            push_constant_offset: Some(pc),
            bound_location: None,
            bound_builtin_io: None,
            workgroup_shared: false,
            ..
        } => {
            if !ty.can_uniform_struct_member() {
                panic!("Error: this type cannot be used as push constant value")
            }

            global_vars
                .push_constants
                .push(ShaderInterfacePushConstantVariable {
                    ty: ty.make_spv_type(scope),
                    original_refpath: refpath.clone(),
                    offset: *pc,
                });
        }
        SymbolAttribute {
            bound_location: Some(loc),
            bound_builtin_io: None,
            workgroup_shared: false,
            ..
        } => global_vars.inputs.push(ShaderInterfaceInputVariable {
            ty: ty.make_spv_type(scope),
            original_refpath: refpath.clone(),
            decorations: vec![spv::Decorate::Location(*loc)],
        }),
        SymbolAttribute {
            bound_builtin_io: Some(b),
            workgroup_shared: false,
            ..
        } => global_vars.inputs.push(ShaderInterfaceInputVariable {
            ty: ty.make_spv_type(scope),
            original_refpath: refpath.clone(),
            decorations: vec![spv::Decorate::Builtin(match b {
                BuiltinInputOutput::Position => spv::asm::Builtin::Position,
                BuiltinInputOutput::VertexID => spv::asm::Builtin::VertexIndex,
                BuiltinInputOutput::InstanceID => spv::asm::Builtin::InstanceIndex,
                BuiltinInputOutput::LocalInvocationIndex => spv::asm::Builtin::LocalInvocationIndex,
                BuiltinInputOutput::GlobalInvocationID => spv::asm::Builtin::GlobalInvocationId,
            })],
        }),
        SymbolAttribute {
            workgroup_shared: true,
            ..
        } => global_vars
            .workgroup_shared_vars
            .push(WorkgroupSharedVariable {
                ty: ty.make_spv_type(scope),
                original_refpath: refpath.clone(),
            }),
        _ => match ty {
            ConcreteType::Struct(members) => {
                for (n, m) in members.iter().enumerate() {
                    process_entry_point_inputs(
                        &m.attribute,
                        &RefPath::Member(Box::new(refpath.clone()), n),
                        &m.ty,
                        scope,
                        global_vars,
                    );
                }
            }
            _ => panic!("Error: non-decorated primitive input found: {:?}", refpath),
        },
    }
}

fn process_entry_point_outputs<'s>(
    attr: &SymbolAttribute,
    ty: &ConcreteType<'s>,
    scope: &SymbolScope<'_, 's>,
    output_variables: &mut Vec<ShaderInterfaceOutputVariable>,
) {
    match attr {
        SymbolAttribute {
            module_entry_point: true,
            ..
        } => panic!("Error: entry point output cannot be a module entry point"),
        SymbolAttribute {
            shader_model: Some(_),
            ..
        } => panic!("Error: entry point output cannot have any shader model attributes"),
        SymbolAttribute {
            descriptor_set_location: Some(_),
            descriptor_set_binding: Some(_),
            push_constant_offset: None,
            bound_location: None,
            bound_builtin_io: None,
            ..
        } => panic!("Error: entry point output cannot have any descriptor set bindings"),
        SymbolAttribute {
            push_constant_offset: Some(_),
            bound_location: None,
            bound_builtin_io: None,
            ..
        } => panic!("Error: entry point output cannot have any push constant binding"),
        SymbolAttribute {
            bound_location: Some(loc),
            bound_builtin_io: None,
            ..
        } => output_variables.push(ShaderInterfaceOutputVariable {
            ty: ty.make_spv_type(scope),
            decorations: vec![spv::Decorate::Location(*loc)],
        }),
        SymbolAttribute {
            bound_builtin_io: Some(b),
            ..
        } => output_variables.push(ShaderInterfaceOutputVariable {
            ty: ty.make_spv_type(scope),
            decorations: vec![spv::Decorate::Builtin(match b {
                BuiltinInputOutput::Position => spv::asm::Builtin::Position,
                BuiltinInputOutput::VertexID => spv::asm::Builtin::VertexIndex,
                BuiltinInputOutput::InstanceID => spv::asm::Builtin::InstanceIndex,
                BuiltinInputOutput::LocalInvocationIndex => {
                    panic!("Error: cannot output LocalInvocationIndex")
                }
                BuiltinInputOutput::GlobalInvocationID => {
                    panic!("Error: cannot output GlobalInvocationID")
                }
            })],
        }),
        _ => match ty {
            ConcreteType::Struct(members) => {
                for m in members {
                    process_entry_point_outputs(&m.attribute, &m.ty, scope, output_variables);
                }
            }
            _ => panic!("Error: non-decorated primitive output found"),
        },
    }
}
