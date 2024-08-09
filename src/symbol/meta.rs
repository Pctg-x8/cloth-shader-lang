use crate::parser::{AttributeArgSyntax, AttributeSyntax, ExpressionNode};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SymbolAttribute {
    pub module_entry_point: bool,
    pub shader_model: Option<ShaderModel>,
    pub descriptor_set_location: Option<u32>,
    pub descriptor_set_binding: Option<u32>,
    pub input_attachment_index: Option<u32>,
    pub push_constant_offset: Option<u32>,
    pub compute_shader_local_size: Option<(u32, u32, u32)>,
    pub bound_location: Option<u32>,
    pub bound_builtin_io: Option<BuiltinInputOutput>,
    pub workgroup_shared: bool,
    pub storage_buffer: bool,
    pub image_format_specifier: Option<crate::spirv::asm::ImageFormat>,
}
impl Default for SymbolAttribute {
    fn default() -> Self {
        Self {
            module_entry_point: false,
            shader_model: None,
            descriptor_set_location: None,
            descriptor_set_binding: None,
            input_attachment_index: None,
            push_constant_offset: None,
            compute_shader_local_size: None,
            bound_location: None,
            bound_builtin_io: None,
            workgroup_shared: false,
            storage_buffer: false,
            image_format_specifier: None,
        }
    }
}
impl SymbolAttribute {
    pub const fn is_empty(&self) -> bool {
        matches!(
            self,
            SymbolAttribute {
                module_entry_point: false,
                shader_model: None,
                descriptor_set_location: None,
                descriptor_set_binding: None,
                input_attachment_index: None,
                push_constant_offset: None,
                compute_shader_local_size: None,
                bound_location: None,
                bound_builtin_io: None,
                workgroup_shared: false,
                storage_buffer: false,
                image_format_specifier: None,
            }
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderModel {
    VertexShader,
    TessellationControlShader,
    TessellationEvaluationShader,
    GeometryShader,
    FragmentShader,
    ComputeShader,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinInputOutput {
    Position,
    VertexID,
    InstanceID,
    LocalInvocationIndex,
    GlobalInvocationID,
}

pub fn eval_symbol_attributes(attr: SymbolAttribute, a: AttributeSyntax) -> SymbolAttribute {
    let args = match a.arg {
        Some(AttributeArgSyntax::Single(x)) => vec![(x, None)],
        Some(AttributeArgSyntax::Multiple { arg_list, .. }) => arg_list,
        None => Vec::new(),
    };

    // TODO: User-defined Attributeとかシンボルエイリアスとかをサポートするようになったら真面目に型解決して処理する必要がある
    match a.name_token.slice {
        "DescriptorSet" => match <&[(ExpressionNode, _); 1]>::try_from(&args[..]) {
            Ok([(ExpressionNode::Number(n), _)]) => SymbolAttribute {
                descriptor_set_location: Some(n.slice.parse().unwrap()),
                ..attr
            },
            Ok(_) => panic!("unsupported or not constant expression"),
            Err(_) => panic!("DescriptorSet attribute requires a number as argument"),
        },
        "Binding" => match <&[(ExpressionNode, _); 1]>::try_from(&args[..]) {
            Ok([(ExpressionNode::Number(n), _)]) => SymbolAttribute {
                descriptor_set_binding: Some(n.slice.parse().unwrap()),
                ..attr
            },
            Ok(_) => panic!("unsupported or not constant expression"),
            Err(_) => panic!("Binding attribute requires a number as argument"),
        },
        "InputAttachment" => match <&[(ExpressionNode, _); 1]>::try_from(&args[..]) {
            Ok([(ExpressionNode::Number(n), _)]) => SymbolAttribute {
                input_attachment_index: Some(n.slice.parse().unwrap()),
                ..attr
            },
            Ok(_) => panic!("unsupported or not constant expression"),
            Err(_) => panic!("InputAttachment attribute requires a number as argument"),
        },
        "PushConstant" => match <&[(ExpressionNode, _); 1]>::try_from(&args[..]) {
            Ok([(ExpressionNode::Number(n), _)]) => SymbolAttribute {
                push_constant_offset: Some(n.slice.parse().unwrap()),
                ..attr
            },
            Ok(_) => panic!("unsupported or not constant expression"),
            Err(_) => panic!("PushConstant attribute requires a number as argument"),
        },
        "Location" => match <&[(ExpressionNode, _); 1]>::try_from(&args[..]) {
            Ok([(ExpressionNode::Number(n), _)]) => SymbolAttribute {
                bound_location: Some(n.slice.parse().unwrap()),
                ..attr
            },
            Ok(_) => panic!("unsupported or not constant expression"),
            Err(_) => panic!("Location attribute requires a number as argument"),
        },
        "Position" => {
            if args.is_empty() {
                SymbolAttribute {
                    bound_builtin_io: Some(BuiltinInputOutput::Position),
                    ..attr
                }
            } else {
                panic!("Position attribute does not take any arguments");
            }
        }
        "VertexID" => {
            if args.is_empty() {
                SymbolAttribute {
                    bound_builtin_io: Some(BuiltinInputOutput::VertexID),
                    ..attr
                }
            } else {
                panic!("VertexID attribute does not take any arguments");
            }
        }
        "InstanceID" => {
            if args.is_empty() {
                SymbolAttribute {
                    bound_builtin_io: Some(BuiltinInputOutput::InstanceID),
                    ..attr
                }
            } else {
                panic!("InstanceID attribute does not take any arguments");
            }
        }
        "LocalInvocationIndex" => {
            if args.is_empty() {
                SymbolAttribute {
                    bound_builtin_io: Some(BuiltinInputOutput::LocalInvocationIndex),
                    ..attr
                }
            } else {
                panic!("LocalInvocationIndex attribute does not take any arguments");
            }
        }
        "GlobalInvocationID" => {
            if args.is_empty() {
                SymbolAttribute {
                    bound_builtin_io: Some(BuiltinInputOutput::GlobalInvocationID),
                    ..attr
                }
            } else {
                panic!("GlobalInvocationID attribute does not take any arguments");
            }
        }
        "VertexShader" => {
            if args.is_empty() {
                SymbolAttribute {
                    module_entry_point: true,
                    shader_model: Some(ShaderModel::VertexShader),
                    ..attr
                }
            } else {
                panic!("VertexShader attribute does not take any arguments");
            }
        }
        "FragmentShader" => {
            if args.is_empty() {
                SymbolAttribute {
                    module_entry_point: true,
                    shader_model: Some(ShaderModel::FragmentShader),
                    ..attr
                }
            } else {
                panic!("VertexShader attribute does not take any arguments");
            }
        }
        "ComputeShader" => match <&[(ExpressionNode, _); 3]>::try_from(&args[..]) {
            Ok(
                [(ExpressionNode::Number(x), _), (ExpressionNode::Number(y), _), (ExpressionNode::Number(z), _)],
            ) => SymbolAttribute {
                module_entry_point: true,
                shader_model: Some(ShaderModel::ComputeShader),
                compute_shader_local_size: Some((
                    x.slice.parse().unwrap(),
                    y.slice.parse().unwrap(),
                    z.slice.parse().unwrap(),
                )),
                ..attr
            },
            Ok(_) => panic!("unsupported or not constant expression"),
            Err(_) => panic!("ComputeShader attribute requires 3 number arguments"),
        },
        "WorkgroupShared" => {
            if args.is_empty() {
                SymbolAttribute {
                    workgroup_shared: true,
                    ..attr
                }
            } else {
                panic!("WorkgroupShared attribute does not take any arguments");
            }
        }
        "StorageBuffer" => {
            if args.is_empty() {
                SymbolAttribute {
                    storage_buffer: true,
                    ..attr
                }
            } else {
                panic!("StorageBuffer attribute does not take any arguments");
            }
        }
        "ImageFormat" => match <&[(ExpressionNode, _); 1]>::try_from(&args[..]) {
            Ok([(ExpressionNode::Var(x), _)]) if x.slice == "Rgba16F" => SymbolAttribute {
                image_format_specifier: Some(crate::spirv::asm::ImageFormat::Rgba16f),
                ..attr
            },
            Ok(_) => panic!("unsupported or not ImageFormat expression"),
            Err(_) => panic!("ComputeShader attribute requires 3 number arguments"),
        },
        _ => panic!("{}: Unknown attribute", a.name_token.slice),
    }
}
