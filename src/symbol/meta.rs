use crate::parser::{AttributeArgSyntax, AttributeSyntax, ConstExpressionNode};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SymbolAttribute {
    pub module_entry_point: bool,
    pub shader_model: Option<ShaderModel>,
    pub descriptor_set_location: Option<u32>,
    pub descriptor_set_binding: Option<u32>,
    pub input_attachment_index: Option<u32>,
    pub push_constant_offset: Option<u32>,
    pub bound_location: Option<u32>,
    pub bound_builtin_io: Option<BuiltinInputOutput>,
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
            bound_location: None,
            bound_builtin_io: None,
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
                bound_location: None,
                bound_builtin_io: None
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
}

pub fn eval_symbol_attributes(attr: SymbolAttribute, a: AttributeSyntax) -> SymbolAttribute {
    let args = match a.arg {
        Some(AttributeArgSyntax::Single(x)) => vec![(x, None)],
        Some(AttributeArgSyntax::Multiple { arg_list, .. }) => arg_list,
        None => Vec::new(),
    };

    // TODO: User-defined Attributeとかシンボルエイリアスとかをサポートするようになったら真面目に型解決して処理する必要がある
    match a.name_token.slice {
        "DescriptorSet" => match <&[(ConstExpressionNode, _); 1]>::try_from(&args[..]) {
            Ok([(ConstExpressionNode::Number(n), _)]) => SymbolAttribute {
                descriptor_set_location: Some(n.slice.parse().unwrap()),
                ..attr
            },
            Err(_) => panic!("DescriptorSet attribute requires a number as argument"),
        },
        "Binding" => match <&[(ConstExpressionNode, _); 1]>::try_from(&args[..]) {
            Ok([(ConstExpressionNode::Number(n), _)]) => SymbolAttribute {
                descriptor_set_binding: Some(n.slice.parse().unwrap()),
                ..attr
            },
            Err(_) => panic!("Binding attribute requires a number as argument"),
        },
        "InputAttachment" => match <&[(ConstExpressionNode, _); 1]>::try_from(&args[..]) {
            Ok([(ConstExpressionNode::Number(n), _)]) => SymbolAttribute {
                input_attachment_index: Some(n.slice.parse().unwrap()),
                ..attr
            },
            Err(_) => panic!("InputAttachment attribute requires a number as argument"),
        },
        "PushConstant" => match <&[(ConstExpressionNode, _); 1]>::try_from(&args[..]) {
            Ok([(ConstExpressionNode::Number(n), _)]) => SymbolAttribute {
                push_constant_offset: Some(n.slice.parse().unwrap()),
                ..attr
            },
            Err(_) => panic!("PushConstant attribute requires a number as argument"),
        },
        "Location" => match <&[(ConstExpressionNode, _); 1]>::try_from(&args[..]) {
            Ok([(ConstExpressionNode::Number(n), _)]) => SymbolAttribute {
                bound_location: Some(n.slice.parse().unwrap()),
                ..attr
            },
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
        _ => panic!("{}: Unknown attribute", a.name_token.slice),
    }
}
