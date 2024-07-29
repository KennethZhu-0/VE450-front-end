document.getElementById('openMoreParamsButton').addEventListener('click', function() {
    document.getElementById('moreParamsModal').style.display = 'flex';
});

document.getElementsByClassName('close')[0].addEventListener('click', function() {
    document.getElementById('moreParamsModal').style.display = 'none';
});

document.getElementById('submitMoreParams').addEventListener('click', function() {
    document.getElementById('moreParamsModal').style.display = 'none';
});


document.getElementById('basicForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    // Collect basic inputs
    const inputs = {
        x: document.getElementById('x').value,
        y: document.getElementById('y').value,
        z: document.getElementById('z').value
    };

    // Collect more inputs if available
    for (let i = 4; i <= 27; i++) {
        const inputElement = document.getElementById(`input${i}`);
        if (inputElement) {
            inputs[`input${i}`] = inputElement.value;
        }
    }

    // Send inputs to backend model
    try {
        const response = await fetch('http://127.0.0.1:5000/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(inputs)
        });

        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }

        const result = await response.json();
        
        // Display output
        document.getElementById('maxForce').value = result[0];
        document.getElementById('nodalDeformationX').value = result[1];
        document.getElementById('nodalDeformationY').value = result[2];
        document.getElementById('nodalDeformationZ').value = result[3];
        document.getElementById('nodalFailure').value = result[4];
        document.getElementById('displacement').value = result[5];
        document.getElementById('force').value = result[5];
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('output').innerText = 'Error: ' + error.message;
    }

});
