import pl from 'nodejs-polars';


const df = pl.readCSV('test.csv');

/* const SLP_value = document.getElementById("SLP_value").value; */

const SLP_value_results = df.filter(pl.col("SLP_values").eq(1000));

const mean_value = SLP_value_results.select(pl.col("mean"))[0][0].toFixed(3);

const p_interval_lower = SLP_value_results.select(pl.col("obs_ci_lower"))[0][0].toFixed(3);

const p_interval_upper = SLP_value_results.select(pl.col("obs_ci_upper"))[0][0].toFixed(3);

console.log(p_interval_lower)

function surge_msg() {

    let input_value = document.getElementById("SLP_value").value;
    let output_element = document.getElementById("result_msg");
    output_element.innerHTML = input_value; 
}

function clear_msg() {

    let output_element = document.getElementById("result_msg");
    output_element.innerHTML = "";
}

let button = document.getElementById("my_button");
button.addEventListener("click", surge_msg);

let clear_btn = document.getElementById("clear_button");
clear_btn.addEventListener("click", clear_msg);


/* console.log("95% confidence the true storm surge value is between " + p_interval_lower + "m - " + p_interval_upper + "m.\n");
console.log("Best guess is " + mean_value + "m.");
 */
