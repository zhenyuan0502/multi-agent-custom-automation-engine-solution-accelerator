(() => {
    const notyf = new Notyf({ position: { x: 'right', y: 'top' }, ripple: false, duration: 3000 });
    const apiEndpoint = sessionStorage.getItem('apiEndpoint');
    const newTaskPrompt = document.getElementById("newTaskPrompt");
    const startTaskButton = document.getElementById("startTaskButton");
    newTaskPrompt.focus();


    const startTask = () => {

        startTaskButton.addEventListener('click', (event) => {

            const sessionId = 'sid_' + (new Date()).getTime() + '_' + Math.floor(Math.random() * 10000);

            newTaskPrompt.disabled = true;
            startTaskButton.disabled = true;
            startTaskButton.classList.add('is-loading');
            
            window.headers
                .then(headers =>{
                    fetch(apiEndpoint + '/input_task', {
                        method: 'POST',
                        headers: headers,
                        body: JSON.stringify({
                            session_id: sessionId,
                            description: newTaskPrompt.value
                        })
                    })
                .then(response => response.json())
                .then(data => {

                    if (data.status == 'Plan not created') {
                        notyf.error('Unable to create plan for this task.');
                        newTaskPrompt.disabled = false;
                        startTaskButton.disabled = false;
                        return;
                    }

                    console.log('startTaskButton', data);

                    newTaskPrompt.disabled = false;
                    startTaskButton.disabled = false;
                    startTaskButton.classList.remove('is-loading');

                    window.parent.postMessage({
                        action: 'taskStarted',
                        session_id: data.session_id,
                        task_id: data.plan_id,
                        task_name: newTaskPrompt.value
                    }, '*');

                    newTaskPrompt.value = '';

                    notyf.success('Task created successfully. AI agents are on it!');

                })
                .catch(error => {
                    console.error('Error:', error);
                    newTaskPrompt.disabled = false;
                    startTaskButton.disabled = false;
                    startTaskButton.classList.remove('is-loading');
                })
        });
    })
    };

    const quickTasks = () => {

        document.querySelectorAll('.quick-task').forEach(task => {
            task.addEventListener('click', (event) => {
                const quickTaskPrompt = task.querySelector('.quick-task-prompt').innerHTML;
                newTaskPrompt.value = quickTaskPrompt.trim().replace(/\s+/g, ' ');
            });
        });

    }
    const handleTextAreaTyping = () => {
        const newTaskPrompt = document.getElementById('newTaskPrompt');
        newTaskPrompt.addEventListener('input', () => {
            // whenever text is sent to the text area, we want to update the character count and dynamically resize the text area
            const textInput = document.getElementById('newTaskPrompt');
            const charCount = document.getElementById('charCount');
            
            // Update character count
            charCount.textContent = textInput.value.length;
          
            // Dynamically adjust height
            textInput.style.height = 'auto';
            textInput.style.height = textInput.scrollHeight + 'px';
        });

    }
    
    startTask();
    quickTasks();
    handleTextAreaTyping();

})();