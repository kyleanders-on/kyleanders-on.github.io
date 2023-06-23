async function main() {
  try {
    let parsed_data = await read_file();

    let button = document.getElementById("calc_button");
    button.addEventListener("click", () => surge_msg(parsed_data, "SPU"));

    let clear_btn = document.getElementById("clear_button");
    clear_btn.addEventListener("click", clear_msg);

    let radio_btn = document.querySelectorAll('input[name="client"]');
    radio_btn.forEach((radio) => {
      radio.addEventListener("change", async () => {
        parsed_data = await read_file();
        let radio_id = radio.id;
        let button = document.getElementById("calc_button");
        button.addEventListener("click", () =>
          surge_msg(parsed_data, radio_id)
        );

        let clear_btn = document.getElementById("clear_button");
        clear_btn.addEventListener("click", clear_msg);
      });
    });
  } catch (error) {
    // Handle the error here
    console.error(error);
  }
}

async function read_file() {
  try {
    let file_path = select_filepath();
    let response = await fetch(file_path);
    if (response.ok) {
      let file_contents = await response.text();
      let parsed_data = Papa.parse(file_contents, { header: true }).data;
      return parsed_data;
    } else {
      throw new Error("Error: " + response.status);
    }
  } catch (error) {
    throw new Error("Error: " + error);
  }
}

function surge_msg(parsed_data, client) {
  // Store SLP user input value
  let input_value = document.getElementById("SLP_value").value;

  // One row of model data associated with a given SLP value
  let idx_value = input_value - 800;
  let output_element = document.getElementById("result_msg");

  if (idx_value < 0 || input_value > 1100) {
    output_element.innerHTML = `Please input a realistic SLP value.`;
  } else {
    let SLP_value_data = parsed_data[idx_value];
    let best_guess = parseFloat(SLP_value_data["mean"]).toFixed(2);
    let p_interval_lower = parseFloat(SLP_value_data["obs_ci_lower"]).toFixed(2);
    let p_interval_upper = parseFloat(SLP_value_data["obs_ci_upper"]).toFixed(2);

    let unit = client == "Delta" ? "m" : "ft";
    output_element.innerHTML = `95% confidence the true storm surge value is between ${p_interval_lower}${unit} and ${p_interval_upper}${unit}.<br/><br/>Best guess is ${best_guess}${unit}.`;
  }
}

function clear_msg() {
  let output_element = document.getElementById("result_msg");
  output_element.innerHTML = "";
  document.getElementById("SLP_value").value = "";
}

function select_filepath() {
  var file_path = document.querySelector('input[name="client"]:checked').value;
  return file_path;
}

main();
