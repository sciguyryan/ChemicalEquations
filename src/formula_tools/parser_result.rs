use core::fmt;
use std::collections::HashMap;

use super::utils;

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

    /// Print the contents of this [`ParserResult`] instance.
    pub fn print(&self) -> String {
        let mut str = String::new();

        for symbol in self.symbols.iter() {
            str.push_str(symbol.0);

            for d in symbol.1.to_string().chars() {
                str.push(utils::map_to_subscript(d));
            }
        }

        if self.charge != 0 {
            str = format!("[{}]", str);

            for d in self.charge.to_string().chars() {
                str.push(utils::map_to_superscript(d));
            }

            if self.charge > 0 {
                str.push(utils::map_to_superscript('+'));
            } else {
                str.push(utils::map_to_superscript('-'));
            }
        }

        str
    }
}

impl fmt::Display for ParserResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.print())
    }
}
