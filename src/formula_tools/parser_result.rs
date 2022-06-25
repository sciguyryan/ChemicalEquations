use std::collections::HashMap;

#[derive(Debug, Default, Eq, PartialEq)]
pub struct ParserResult {
    pub symbols: HashMap<String, u32>,
    pub charge: i32,
}

impl ParserResult {
    #[allow(unused)]
    pub fn new() -> Self {
        Self {
            symbols: HashMap::new(),
            charge: 0,
        }
    }
}
