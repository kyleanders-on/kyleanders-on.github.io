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
  console.log(client);

  // Store SLP user input value
  let input_value = document.getElementById("SLP_value").value;

  // One row of model data associated with a given SLP value
  let SLP_value_data = parsed_data[input_value - 800];

  // Separate desired variables from row
  let best_guess = parseFloat(SLP_value_data["mean"]).toFixed(2);
  let p_interval_lower = parseFloat(SLP_value_data["obs_ci_lower"]).toFixed(2);
  let p_interval_upper = parseFloat(SLP_value_data["obs_ci_upper"]).toFixed(2);

  let output_element = document.getElementById("result_msg");

  // if statement that checks for input ID?

  if (client == "Delta") {
    output_element.innerHTML =
      `95% confidence the true storm surge value is between ` +
      p_interval_lower +
      `m - ` +
      p_interval_upper +
      `m.<br/><br/>Best guess is ` +
      best_guess +
      `m.`;
  } else {
    output_element.innerHTML =
      `95% confidence the true storm surge value is between ` +
      p_interval_lower +
      `ft - ` +
      p_interval_upper +
      `ft.<br/><br/>Best guess is ` +
      best_guess +
      `ft.`;
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
