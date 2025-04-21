mod bayes;
mod metrics;
mod tokenizer;
use std::{env, error::Error};

use bayes::{BernouliNB, MultinomialNB, NaiveBayesClassifier};
use parquet::{
    file::{reader::FileReader, serialized_reader::SerializedFileReader},
    record::RowAccessor,
};
use tokenizer::Tokenizer;

#[derive(Debug)]
enum Model {
    Bernoulli,
    Multinomial,
}

fn create_model(model: Model, n_features: usize) -> Box<dyn NaiveBayesClassifier> {
    match model {
        Model::Bernoulli => Box::new(BernouliNB::new(n_features, 2, 0.1)),
        Model::Multinomial => Box::new(MultinomialNB::new(n_features, 2, 0.1)),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let used_model = env::args()
        .nth(1)
        .map(|x| {
            if x == "bernoulli" {
                Model::Bernoulli
            } else {
                Model::Multinomial
            }
        })
        .unwrap_or(Model::Multinomial);

    println!("Using NB {:?} classifier", used_model);

    // Create the tokenizer
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

    // Create the classifier based on provided program arguments
    let mut nb = create_model(used_model, toknzr.token_count());
    
    // Train the classifier on the training set
    training_pairs
        .iter()
        .for_each(|row| nb.fit(&toknzr.tokenize_sparse(&row.0), row.1));

    let eval_predicted = eval_pairs
        .iter()
        .map(|row| nb.predict(&toknzr.tokenize_sparse(&row.0)))
        .collect::<Vec<usize>>();

    let eval_labels = eval_pairs.iter().map(|x| x.1).collect::<Vec<usize>>();

    let confusion_matrix = metrics::confusion_matrix(&eval_predicted[0..], &eval_labels[0..], 2);

    println!("Eval. accuracy: {:.3}", confusion_matrix.accuracy());
    println!("Eval. recall: {:.3}", confusion_matrix.recall(1));

    Ok(())
}
