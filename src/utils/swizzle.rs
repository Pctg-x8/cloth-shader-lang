pub const fn swizzle_indices(x: &str, src_component_count: u8) -> Option<[Option<usize>; 4]> {
    match x.as_bytes() {
        &[a] => Some([swizzle_index(a, src_component_count), None, None, None]),
        &[a, b] => Some([
            swizzle_index(a, src_component_count),
            swizzle_index(b, src_component_count),
            None,
            None,
        ]),
        &[a, b, c] => Some([
            swizzle_index(a, src_component_count),
            swizzle_index(b, src_component_count),
            swizzle_index(c, src_component_count),
            None,
        ]),
        &[a, b, c, d] => Some([
            swizzle_index(a, src_component_count),
            swizzle_index(b, src_component_count),
            swizzle_index(c, src_component_count),
            swizzle_index(d, src_component_count),
        ]),
        _ => None,
    }
}

pub const fn swizzle_index(x: u8, src_component_count: u8) -> Option<usize> {
    match x {
        b'r' | b'R' | b'x' | b'X' if src_component_count >= 1 => Some(0),
        b'g' | b'G' | b'y' | b'Y' if src_component_count >= 2 => Some(1),
        b'b' | b'B' | b'z' | b'Z' if src_component_count >= 3 => Some(2),
        b'a' | b'A' | b'w' | b'W' if src_component_count >= 4 => Some(3),
        _ => None,
    }
}
