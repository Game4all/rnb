mod bayes;
mod metrics;
mod tokenizer;
use std::error::Error;

use bayes::{BernouliNB, MultinomialNB};
use parquet::{
    file::{reader::FileReader, serialized_reader::SerializedFileReader},
    record::RowAccessor,
};
use tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn Error>> {
    let mut toknzr = Tokenizer::new("([.,!?;:=()\"'\\[\\]1234567890/@#*‘&_])");

    // Parse the whole dataset and store it in memory
    let dataset_file = std::fs::File::open("datasets/sms_spam.parquet")?;
    let dataset: SerializedFileReader<std::fs::File> = SerializedFileReader::new(dataset_file)?;

    let mut training_pairs: Vec<(String, usize)> = Vec::new();
    for row in dataset.get_row_iter(None)?.flatten() {
        training_pairs.push((
            row.get_string(0)?.replace("\n", ""),
            row.get_long(1)? as usize,
        ));
    }

    // Split the dataset into a training and eval set
    let eval_pairs: Vec<(String, usize)> = training_pairs.split_off(training_pairs.len() - 100);

    // Fit the tokenizer on every training text pair
    training_pairs.iter().for_each(|row| _ = toknzr.fit(&row.0));
    toknzr.save_to_file(&mut std::fs::File::create("tokenizer.json")?)?;

    println!("Tokenizer vocab size: {}", toknzr.token_count());

    // Train the classifier on the training set
    // let mut nb = BernouliNB::new(toknzr.token_count(), 2, 0.1);
    let mut nb = MultinomialNB::new(toknzr.token_count(), 2, 0.1);
    training_pairs
        .iter()
        .for_each(|row| nb.fit(&toknzr.tokenize_sparse(&row.0), row.1));

    let eval_predicted = eval_pairs
        .iter()
        .map(|row| nb.predict(&toknzr.tokenize_sparse(&row.0)))
        .collect::<Vec<usize>>();

    let eval_labels = eval_pairs.iter().map(|x| x.1).collect::<Vec<usize>>();

    let confusion_matrix = metrics::confusion_matrix(&eval_predicted[0..], &eval_labels[0..], 2);

    println!("Eval. accuracy: {}", confusion_matrix.accuracy());

    Ok(())
}
