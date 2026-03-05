import os

import streamlit as st


@st.fragment(run_every="2s")
def monitor_active_job(job_manager, config):
    """Auto-refreshing fragment to monitor training progress without blinking the whole app."""
    if not st.session_state.get("active_job_id"):
        return

    active_job = job_manager.get_status(st.session_state.active_job_id)
    if not active_job:
        return

    # If job just finished during this fragment execution, trigger a full rerun
    if active_job["status"] in ["completed", "failed"]:
        st.rerun()
 
    if active_job["status"] == "created":
        st.info("⏳ Job created, waiting for worker...")

    elif active_job["status"] == "running":
        st.subheader("🔄 Optimization in Progress")

        progress = active_job.get("progress", 0)
        curr_gen = active_job.get("current_generation", 0)
        total_gen = active_job.get("total_generations", config.get("n_generations", 50))
        message = active_job.get("message", "")
        best_f1 = active_job.get("best_f1", 0.0)
        cache_hit_rate = active_job.get("cache_hit_rate", 0.0)
        total_evaluated = active_job.get("total_evaluated", 0)

        # Progress bar
        st.progress(progress / 100)
        st.text(f"Status: {message}")

        # Live metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("🧬 Generation", f"{curr_gen}/{total_gen}")
        with m2:
            st.metric("🏆 Best F1", f"{best_f1:.4f}")
        with m3:
            st.metric("♻️ Cache Hit Rate", f"{cache_hit_rate:.1f}%")
        with m4:
            st.metric("📊 Unique Pipelines", f"{total_evaluated}")

        if st.button(
            "Stop Monitoring (Job continues in background)", key="stop_monitor_btn"
        ):
            st.session_state.active_job_id = None
            st.rerun()

    # Terminal-style Log Streamer logic within fragment
    if active_job["status"] in ["created", "running"]:
        with st.expander(
            "📝 Live Log Stream", expanded=(active_job["status"] == "running")
        ):
            log_file = f"logs/run_{st.session_state.active_job_id}.log"
            if os.path.exists(log_file):
                st.caption(f"Showing logs for job: {st.session_state.active_job_id}")
                with open(log_file, "r") as f:
                    lines = f.readlines()[-100:]
                    st.code("".join(lines), language="log")
            else:
                st.info("Waiting for logs...")


def run_automl(job_manager):
    from ui.layout import (
        render_config,
    )

    config = render_config()

    # Quick Demo Mode
    st.markdown("---")
    st.subheader("⚡ Quick Demo Mode")
    quick_demo = st.checkbox(
        "Enable Quick Demo (3 minutes)",
        value=False,
        help="Fast configuration for quick testing: 1K samples, 10 pop, 5 gen, 10 BO",
    )

    if quick_demo:
        config["max_samples"] = 1000
        config["population_size"] = 10
        config["n_generations"] = 5
        config["bo_calls"] = 10
        st.success("✓ Quick demo mode enabled!")
        st.write(f"Est. runtime: ~3-5 minutes")

    # Check for active job status
    active_job = None
    if st.session_state.active_job_id:
        active_job = job_manager.get_status(st.session_state.active_job_id)
        if active_job and active_job["status"] in ["completed", "failed"]:
            # If job just finished, load results
            if active_job["status"] == "completed" and st.session_state.results is None:
                st.success("🎉 Job completed successfully!")
                results = job_manager.get_result(st.session_state.active_job_id)
                if results:
                    st.session_state.results = results
            elif active_job["status"] == "failed":
                st.error(f"❌ Job failed: {active_job.get('error', 'Unknown error')}")

    # Run button
    # Disable run button if job is running
    is_running = (active_job is not None) and (active_job.get("status") == "running")
    run_button = st.button("🚀 Run AutoML", type="primary", disabled=is_running)

    if is_running:
        st.info(f"🔄 Job {st.session_state.active_job_id} is running...")
    elif st.session_state.results is not None:
        st.success("✅ Results available - view them in the 'History & Analysis' tab")

    if run_button:
        # Clear previous results
        st.session_state.results = None
        st.session_state.baseline_results = None
        st.session_state.active_job_id = None

        # Create new job
        with st.spinner("Initializing job..."):
            job_id = job_manager.create_job(config)
            st.session_state.active_job_id = job_id
            st.rerun()

    # Job Progress Monitoring (Auto-refreshing fragment)
    if active_job and active_job["status"] in ["created", "running"]:
        monitor_active_job(job_manager, config)
