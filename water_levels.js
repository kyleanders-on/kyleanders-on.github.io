/* import pl from 'nodejs-polars';


const df = pl.readCSV('test.csv');

//const SLP_value = document.getElementById("SLP_value").value; 

const SLP_value_results = df.filter(pl.col("SLP_values").eq(1000));

const mean_value = SLP_value_results.select(pl.col("mean"))[0][0].toFixed(3);

const p_interval_lower = SLP_value_results.select(pl.col("obs_ci_lower"))[0][0].toFixed(3);

const p_interval_upper = SLP_value_results.select(pl.col("obs_ci_upper"))[0][0].toFixed(3);

console.log(p_interval_lower) */

function surge_msg() {

    let input_value = document.getElementById("SLP_value").value;
    let output_element = document.getElementById("result_msg");
    //output_element.innerHTML = input_value; 
    output_element.innerHTML = "95% confidence the true storm surge value is between 0.234m - 0.431m.<br/><br/>Best guess is 0.311m.";
}

function clear_msg() {

    let output_element = document.getElementById("result_msg");
    output_element.innerHTML = "";
}

let button = document.getElementById("my_button");
button.addEventListener("click", surge_msg);

let clear_btn = document.getElementById("clear_button");
clear_btn.addEventListener("click", clear_msg);


async function read_file() {
    try {
        const response = await fetch('https://github.com/marbzgrop/marbzgrop.github.io/blob/main/test.csv');
        if (response.ok) {
            const file_contents = await response.text();
            console.log(file_contents);
        } else {
            console.error('Error: ' + response.status);
        }
    } catch (error) {
        console.error('Error: ' + error);
    }
}

read_file();
