// API calls to OpenEnv endpoints

function toggleTimeInputs() {
    const actionType = document.getElementById("actionType").value;
    const timeInputs = document.getElementById("timeInputs");
    if (actionType === "reschedule") {
        timeInputs.style.display = "block";
    } else {
        timeInputs.style.display = "none";
    }
}

function showToast(msg, isError=false) {
    const toast = document.getElementById("toast");
    toast.className = `toast ${isError ? 'error' : 'success'}`;
    toast.innerText = msg;
    setTimeout(() => {
        toast.className = "toast hidden";
    }, 3000);
}

async function resetEnv() {
    const taskName = document.getElementById("taskName").value;
    try {
        const res = await fetch("/reset", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({task_name: taskName, seed: 42})
        });
        const data = await res.json();
        updateUI(data);
        showToast("Environment Reset Successfully", false);
    } catch (e) {
        showToast("Error resetting environment", true);
    }
}

async function stepEnv() {
    const meetingId = document.getElementById("meetingId").value;
    const actionType = document.getElementById("actionType").value;
    
    if (!meetingId) {
        showToast("Please enter a Meeting ID", true);
        return;
    }

    let action = {
        meeting_id: meetingId,
        action_type: actionType
    };

    if (actionType === "reschedule") {
        action.new_start_hour = parseInt(document.getElementById("newHour").value);
        action.new_start_minute = parseInt(document.getElementById("newMinute").value);
    }

    try {
        const res = await fetch("/step", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({action: action})
        });
        const data = await res.json();
        updateUI(data);
        
        if (data.last_action_error) {
            showToast(data.last_action_error, true);
        } else if (data.done) {
            showToast("Episode Complete! Final Reward: " + data.reward, false);
        } else {
            showToast("Action Accepted", false);
        }
    } catch (e) {
        showToast("Error taking action", true);
    }
}

function updateUI(data) {
    if (!data.observation) return;
    const obs = data.observation;
    
    document.getElementById("conflictCount").innerText = obs.num_conflicts || 0;
    document.getElementById("rewardScore").innerText = (data.reward || 0).toFixed(2);
    document.getElementById("currentTaskBadge").innerText = "Task: " + (obs.task_name || "easy");

    const grid = document.getElementById("calendarGrid");
    grid.innerHTML = "";

    // Identify conflicts map
    const conflictIds = new Set();
    if (obs.active_conflicts) {
        obs.active_conflicts.forEach(c => {
            conflictIds.add(c.meeting_a_id);
            conflictIds.add(c.meeting_b_id);
        });
    }

    if (obs.current_schedule) {
        // sort by start time
        const schedule = obs.current_schedule.sort((a,b) => {
            const timeA = a.time_slot.start_hour * 60 + a.time_slot.start_minute;
            const timeB = b.time_slot.start_hour * 60 + b.time_slot.start_minute;
            return timeA - timeB;
        });

        schedule.forEach(m => {
            const card = document.createElement("div");
            card.className = "meeting-card";
            if (m.is_locked) card.classList.add("meeting-locked");
            if (conflictIds.has(m.meeting_id)) card.classList.add("meeting-conflict");

            const endHour = Math.floor((m.time_slot.start_hour * 60 + m.time_slot.start_minute + m.time_slot.duration_min) / 60);
            const endMin = (m.time_slot.start_minute + m.time_slot.duration_min) % 60;
            
            const timeStr = `${m.time_slot.start_hour.toString().padStart(2, '0')}:${m.time_slot.start_minute.toString().padStart(2, '0')} - ${endHour.toString().padStart(2, '0')}:${endMin.toString().padStart(2, '0')}`;

            let attendees = m.attendees ? m.attendees.join(", ") : "";

            card.innerHTML = `
                <div class="meeting-info">
                    <h4>[${m.meeting_id}] ${m.title}</h4>
                    <p>${attendees} | Priority: ${m.priority}</p>
                </div>
                <div class="meeting-time">
                    ${timeStr} ${m.is_locked ? " 🔒" : ""}
                </div>
            `;
            grid.appendChild(card);
        });
    }

    const constList = document.getElementById("constraintsList");
    constList.innerHTML = "";
    if (obs.constraints) {
        obs.constraints.forEach(c => {
            const li = document.createElement("li");
            li.innerText = `${c.constraint_type} on ${c.target}`;
            constList.appendChild(li);
        });
    }
}

// Initial pull to get state if possible, or just reset
window.onload = () => {
    resetEnv();
};
