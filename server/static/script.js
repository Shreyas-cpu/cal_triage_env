const state = {
    taskName: "easy",
    stepCount: 0,
    lastEpisodeId: "-",
    tasks: []
};

const el = {
    taskName: document.getElementById("taskName"),
    actionType: document.getElementById("actionType"),
    meetingId: document.getElementById("meetingId"),
    newHour: document.getElementById("newHour"),
    newMinute: document.getElementById("newMinute"),
    timeInputs: document.getElementById("timeInputs"),
    conflictCount: document.getElementById("conflictCount"),
    rewardScore: document.getElementById("rewardScore"),
    stepCount: document.getElementById("stepCount"),
    episodeStatus: document.getElementById("episodeStatus"),
    healthPill: document.getElementById("healthPill"),
    taskPill: document.getElementById("taskPill"),
    scheduleMeta: document.getElementById("scheduleMeta"),
    scheduleList: document.getElementById("scheduleList"),
    conflictsList: document.getElementById("conflictsList"),
    constraintsList: document.getElementById("constraintsList"),
    tasksList: document.getElementById("tasksList"),
    eventFeed: document.getElementById("eventFeed"),
    gradeOutput: document.getElementById("gradeOutput"),
    stateHint: document.getElementById("stateHint"),
    infoTotalMeetings: document.getElementById("infoTotalMeetings"),
    infoLockedMeetings: document.getElementById("infoLockedMeetings"),
    infoConflictRate: document.getElementById("infoConflictRate"),
    meetingsBar: document.getElementById("meetingsBar"),
    lockedBar: document.getElementById("lockedBar"),
    conflictBar: document.getElementById("conflictBar"),
    toast: document.getElementById("toast"),
    resetBtn: document.getElementById("resetBtn"),
    stateBtn: document.getElementById("stateBtn"),
    stepBtn: document.getElementById("stepBtn"),
    tasksBtn: document.getElementById("tasksBtn"),
    gradeBtn: document.getElementById("gradeBtn")
};

function pad2(v) {
    return String(v).padStart(2, "0");
}

function endTime(slot) {
    const total = slot.start_hour * 60 + slot.start_minute + slot.duration_min;
    return {
        h: Math.floor(total / 60),
        m: total % 60
    };
}

function logEvent(msg) {
    const li = document.createElement("li");
    const now = new Date();
    li.textContent = `${pad2(now.getHours())}:${pad2(now.getMinutes())}:${pad2(now.getSeconds())} - ${msg}`;
    el.eventFeed.prepend(li);
    while (el.eventFeed.children.length > 12) {
        el.eventFeed.removeChild(el.eventFeed.lastChild);
    }
}

function showToast(msg, isError = false) {
    el.toast.className = `toast ${isError ? "err" : "ok"}`;
    el.toast.textContent = msg;
    window.setTimeout(() => {
        el.toast.className = "toast hidden";
    }, 2400);
}

function toggleTimeInputs() {
    const show = el.actionType.value === "reschedule";
    el.timeInputs.classList.toggle("hidden", !show);
}

function setHealth(ok) {
    el.healthPill.innerHTML = `<span class="dot"></span> ${ok ? "API healthy" : "API unavailable"}`;
    if (!ok) {
        el.healthPill.style.borderColor = "#ff5a5a";
    } else {
        el.healthPill.style.borderColor = "";
    }
}

async function api(path, options = {}) {
    const res = await fetch(path, options);
    let payload = {};
    try {
        payload = await res.json();
    } catch (_) {
        payload = {};
    }
    if (!res.ok) {
        const message = payload.detail || `Request failed: ${res.status}`;
        throw new Error(message);
    }
    return payload;
}

function renderChips(node, items, mapper) {
    node.innerHTML = "";
    if (!items || items.length === 0) {
        const li = document.createElement("li");
        li.textContent = "none";
        node.appendChild(li);
        return;
    }
    items.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = mapper(item);
        node.appendChild(li);
    });
}

function renderSchedule(schedule = [], conflictIds = new Set()) {
    el.scheduleList.innerHTML = "";
    const sorted = [...schedule].sort((a, b) => {
        const tA = a.time_slot.start_hour * 60 + a.time_slot.start_minute;
        const tB = b.time_slot.start_hour * 60 + b.time_slot.start_minute;
        return tA - tB;
    });

    el.scheduleMeta.textContent = `${sorted.length} meetings`;

    sorted.forEach((m) => {
        const end = endTime(m.time_slot);
        const item = document.createElement("article");
        item.className = "meeting";
        item.innerHTML = `
            <div class="title-row">
                <span class="name">[${m.meeting_id}] ${m.title}</span>
                <span class="badge">${m.priority}</span>
            </div>
            <div class="meta">
                <span>${pad2(m.time_slot.start_hour)}:${pad2(m.time_slot.start_minute)}-${pad2(end.h)}:${pad2(end.m)}</span>
                <span>${(m.attendees || []).join(", ") || "no attendees"}</span>
                ${m.is_locked ? "<span class='badge'>locked</span>" : ""}
                ${conflictIds.has(m.meeting_id) ? "<span class='badge'>conflict</span>" : ""}
            </div>
        `;
        el.scheduleList.appendChild(item);
    });
}

function applyTransitionData(data) {
    if (!data || !data.observation) return;
    const obs = data.observation;
    const schedule = obs.current_schedule || [];

    state.taskName = obs.task_name || state.taskName;
    state.stepCount += 1;
    state.lastEpisodeId = data.episode_id || state.lastEpisodeId;

    el.conflictCount.textContent = String(obs.num_conflicts || 0);
    el.rewardScore.textContent = Number(data.reward || 0).toFixed(2);
    el.stepCount.textContent = String(obs.step_count ?? state.stepCount);
    el.episodeStatus.textContent = data.done ? "DONE" : "RUNNING";
    el.taskPill.textContent = `Task: ${state.taskName}`;
    el.stateHint.textContent = `ep=${state.lastEpisodeId}`;

    const conflictIds = new Set();
    (obs.active_conflicts || []).forEach((c) => {
        if (c.meeting_a_id) conflictIds.add(c.meeting_a_id);
        if (c.meeting_b_id) conflictIds.add(c.meeting_b_id);
    });

    renderSchedule(schedule, conflictIds);

    const totalMeetings = schedule.length;
    const lockedMeetings = schedule.filter((m) => m.is_locked).length;
    const lockedRatio = totalMeetings > 0 ? Math.round((lockedMeetings / totalMeetings) * 100) : 0;
    const conflictRate = totalMeetings > 0 ? Math.min(100, Math.round(((obs.num_conflicts || 0) / totalMeetings) * 100)) : 0;
    const meetingsLevel = Math.min(100, totalMeetings * 9);

    if (el.infoTotalMeetings) el.infoTotalMeetings.textContent = String(totalMeetings);
    if (el.infoLockedMeetings) el.infoLockedMeetings.textContent = `${lockedRatio}%`;
    if (el.infoConflictRate) el.infoConflictRate.textContent = `${conflictRate}%`;
    if (el.meetingsBar) el.meetingsBar.style.width = `${meetingsLevel}%`;
    if (el.lockedBar) el.lockedBar.style.width = `${lockedRatio}%`;
    if (el.conflictBar) el.conflictBar.style.width = `${conflictRate}%`;

    renderChips(el.conflictsList, obs.active_conflicts || [], (c) => {
        const a = c.meeting_a_id || "?";
        const b = c.meeting_b_id || "?";
        return `${a} x ${b}`;
    });

    renderChips(el.constraintsList, obs.constraints || [], (c) => {
        const type = c.constraint_type || "constraint";
        const target = c.target || "schedule";
        return `${type}:${target}`;
    });
}

async function checkHealth() {
    try {
        await api("/health");
        setHealth(true);
    } catch (_) {
        setHealth(false);
    }
}

async function resetEnv() {
    state.stepCount = 0;
    const taskName = el.taskName.value;
    try {
        const payload = { task_name: taskName, seed: 42 };
        const data = await api("/reset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        applyTransitionData(data);
        showToast("Environment reset", false);
        logEvent(`reset task=${taskName}`);
    } catch (err) {
        showToast(err.message || "Reset failed", true);
        logEvent("reset failed");
    }
}

async function stepEnv() {
    const meetingId = el.meetingId.value.trim();
    const actionType = el.actionType.value;

    if (!meetingId) {
        showToast("Meeting ID is required", true);
        return;
    }

    const action = {
        meeting_id: meetingId,
        action_type: actionType
    };

    if (actionType === "reschedule") {
        action.new_start_hour = Number(el.newHour.value);
        action.new_start_minute = Number(el.newMinute.value);
    }

    try {
        const data = await api("/step", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action })
        });
        applyTransitionData(data);

        if (data.last_action_error) {
            showToast(data.last_action_error, true);
            logEvent(`step rejected id=${meetingId}`);
            return;
        }

        if (data.done) {
            showToast(`Episode done: reward ${Number(data.reward || 0).toFixed(2)}`, false);
            logEvent(`episode complete reward=${Number(data.reward || 0).toFixed(2)}`);
        } else {
            showToast("Action applied", false);
            logEvent(`step action=${actionType} id=${meetingId}`);
        }
    } catch (err) {
        showToast(err.message || "Step failed", true);
        logEvent("step failed");
    }
}

async function loadState() {
    try {
        const data = await api("/state");
        applyTransitionData(data);
        showToast("State loaded", false);
        logEvent("state refresh");
    } catch (err) {
        showToast(err.message || "State fetch failed", true);
    }
}

async function loadTasks() {
    try {
        const data = await api("/tasks");
        state.tasks = data.tasks || [];
        renderChips(el.tasksList, state.tasks, (t) => `${t.id}:${t.name}`);
        showToast("Tasks loaded", false);
        logEvent(`tasks=${state.tasks.length}`);
    } catch (err) {
        showToast(err.message || "Task fetch failed", true);
    }
}

async function gradeCurrentTask() {
    const taskId = `task_${state.taskName}`;
    try {
        const data = await api(`/grade/${taskId}`);
        const score = Number(data.average_score || 0).toFixed(3);
        const reward = Number(data.average_reward || 0).toFixed(3);
        el.gradeOutput.textContent = `task_id=${taskId}\navg_score=${score}\navg_reward=${reward}\nepisodes=${data.num_episodes || 0}`;
        showToast("Grading complete", false);
        logEvent(`grade ${taskId} score=${score}`);
    } catch (err) {
        el.gradeOutput.textContent = err.message || "grading failed";
        showToast("Grading failed", true);
        logEvent(`grade failed for ${taskId}`);
    }
}

function bindEvents() {
    el.actionType.addEventListener("change", toggleTimeInputs);
    el.taskName.addEventListener("change", () => {
        state.taskName = el.taskName.value;
        el.taskPill.textContent = `Task: ${state.taskName}`;
    });
    el.resetBtn.addEventListener("click", resetEnv);
    el.stepBtn.addEventListener("click", stepEnv);
    el.stateBtn.addEventListener("click", loadState);
    el.tasksBtn.addEventListener("click", loadTasks);
    el.gradeBtn.addEventListener("click", gradeCurrentTask);
}

async function bootstrap() {
    bindEvents();
    toggleTimeInputs();
    await checkHealth();
    await loadTasks();
    await resetEnv();
}

window.addEventListener("load", bootstrap);
