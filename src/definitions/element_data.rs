use lazy_static::lazy_static;
use rusqlite::Connection;
use std::collections::HashMap;

lazy_static! {
    pub static ref ELEMENT_DATA: AllElementData = { AllElementData::new() };
}

#[derive(Debug)]
pub struct ElementData {
    pub atomic_number: i32,
    pub symbol: String,
    pub name: String,
    pub atomic_weight: f32,
}

pub struct AllElementData {
    pub data: HashMap<String, ElementData>,
}

impl AllElementData {
    pub fn new() -> Self {
        let mut aed = Self {
            data: HashMap::new(),
        };

        aed.initialize();

        aed
    }

    fn initialize(&mut self) {
        // TODO: add some error handling here.
        let conn = Connection::open("data.db").expect("Failed to open connection");

        let mut stmt = conn
            .prepare("SELECT atomic_number, symbol, name, atomic_weight FROM Elements")
            .expect("Failed to prepare statement");

        let element_iter = stmt
            .query_map([], |row| {
                Ok(ElementData {
                    atomic_number: row.get(0)?,
                    symbol: row.get(1)?,
                    name: row.get(2)?,
                    atomic_weight: row.get(3)?,
                })
            })
            .expect("Failed to run query");

        for element in element_iter.flatten() {
            self.data.insert(element.symbol.clone(), element);
        }
    }
}
