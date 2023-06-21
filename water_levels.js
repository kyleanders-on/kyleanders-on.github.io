function read_file() {
  return new Promise(async (resolve, reject) => {
    try {
      const response = await fetch(
        'https://raw.githubusercontent.com/marbzgrop/marbzgrop.github.io/main/test.csv'
      );
      if (response.ok) {
        const file_contents = await response.text();
        const parsed_data = Papa.parse(file_contents, { header: true }).data;
        resolve(parsed_data);
      } else {
        reject('Error: ' + response.status);
      }
    } catch (error) {
      reject('Error: ' + error);
    }
  });
}
  


function surge_msg() {


  read_file()
  .then(parsed_data => {

    // Perform further operations on the parsed_data here
    
    // Store SLP user input value
    let input_value = document.getElementById("SLP_value").value;

    // One row of model data associated with a given SLP value
    let SLP_value_data = parsed_data[input_value - 800];

    // Separate desired variables from row
    let best_guess = parseFloat(SLP_value_data['mean']).toFixed(2);
    let p_interval_lower = parseFloat(SLP_value_data['obs_ci_lower']).toFixed(2);
    let p_interval_upper = parseFloat(SLP_value_data['obs_ci_upper']).toFixed(2);

    let output_element = document.getElementById("result_msg");
    //output_element.innerHTML = input_value; 
    output_element.innerHTML = `95% confidence the true storm surge value is between `
     + p_interval_lower + `m - ` + p_interval_upper + `m.<br/><br/>Best guess is ` + best_guess + `m.`;

    
  })
  .catch(error => {
    console.error(error);
  });

}


function clear_msg() {

  let output_element = document.getElementById("result_msg");
  output_element.innerHTML = "";
  document.getElementById("SLP_value").value = '';

}

let button = document.getElementById("my_button");
button.addEventListener("click", surge_msg);

let clear_btn = document.getElementById("clear_button");
clear_btn.addEventListener("click", clear_msg);
