import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from io import StringIO # For reading CSV string
import itertools
import math # Added for sqrt in r-critical calculation

# Helper function to create APA style p-value string
def apa_p_value(p_val):
    if not isinstance(p_val, (int, float)) or np.isnan(p_val):
        return "p N/A"
    try:
        p_val_float = float(p_val)
        if p_val_float < 0.001:
            return "p < .001"
        else:
            return f"p = {p_val_float:.3f}"
    except (ValueError, TypeError):
        return "p N/A (format err)"

# Helper function to format critical values or p-values for display
def format_value_for_display(value, decimals=3, default_str="N/A"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default_str
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return str(value) if value is not None else default_str

def get_dynamic_df_window(all_df_options, selected_df_val, window_size=5):
    """
    Creates a window of df values around the selected_df_val.
    all_df_options: List of all possible df values (can include 'z (∞)' or be numeric).
    selected_df_val: The df value selected by the user (can be 'z (∞)' or numeric).
    window_size: Number of rows to show above and below the selected_df_val.
    Returns a list of df values for the table rows.
    """
    try:
        # Convert 'z (∞)' to a comparable large number for finding index
        temp_selected_df = float('inf') if selected_df_val == 'z (∞)' else float(selected_df_val)
        
        closest_idx = -1
        min_diff = float('inf')

        comparable_options = []
        for option in all_df_options:
            if option == 'z (∞)':
                comparable_options.append(float('inf'))
            # Check if option can be converted to float, including negative numbers and decimals
            elif isinstance(option, (int, float)) or \
                 (isinstance(option, str) and option.replace('.', '', 1).replace('-', '', 1).isdigit()):
                comparable_options.append(float(option))
            else: 
                comparable_options.append(float('-inf')) # Should not be reached if all_df_options are well-formed

        for i, option_val_numeric in enumerate(comparable_options):
            diff = abs(option_val_numeric - temp_selected_df)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
            elif diff == min_diff and option_val_numeric == temp_selected_df: # Prefer exact match
                closest_idx = i
        
        if closest_idx == -1: 
            if selected_df_val in all_df_options:
                try:
                    closest_idx = all_df_options.index(selected_df_val)
                except ValueError: 
                    return all_df_options[:min(len(all_df_options), window_size*2+1)]
            else: 
                return all_df_options[:min(len(all_df_options), window_size*2+1)]

        start_idx = max(0, closest_idx - window_size)
        end_idx = min(len(all_df_options), closest_idx + window_size + 1)
        
        windowed_df_options = all_df_options[start_idx:end_idx]
        
        return windowed_df_options
    except Exception: 
        return all_df_options[:min(len(all_df_options), window_size*2+1)]


# --- Tab 1: t-Distribution ---
def tab_t_distribution():
    st.header("t-Distribution Explorer")
    col1, col2 = st.columns([2, 1.5]) 

    with col1:
        st.subheader("Inputs")
        alpha_t_input = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_t_input")
        
        all_df_values_t = list(range(1, 31)) + [35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 1000, 'z (∞)']
        df_t_selected_display = st.selectbox("Degrees of Freedom (df)", options=all_df_values_t, index=all_df_values_t.index(10), key="df_t_selectbox") 

        if df_t_selected_display == 'z (∞)':
            df_t_calc = np.inf 
        else:
            df_t_calc = int(df_t_selected_display)

        tail_t = st.radio("Tail Selection", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_t_radio")
        test_stat_t = st.number_input("Calculated t-statistic", value=0.0, format="%.3f", key="test_stat_t_input")

        st.subheader("Distribution Plot")
        fig_t, ax_t = plt.subplots(figsize=(8,5)) 
        
        if np.isinf(df_t_calc): 
            dist_label_plot = 'Standard Normal (z)'
            crit_func_ppf_plot = stats.norm.ppf
            crit_func_pdf_plot = stats.norm.pdf
            std_dev_plot = 1.0
        else: 
            dist_label_plot = f't-distribution (df={df_t_calc})'
            crit_func_ppf_plot = lambda q_val: stats.t.ppf(q_val, df_t_calc)
            crit_func_pdf_plot = lambda x_val: stats.t.pdf(x_val, df_t_calc)
            std_dev_plot = stats.t.std(df_t_calc) if df_t_calc > 0 and not np.isinf(df_t_calc) else 1.0
        
        plot_min_t = min(crit_func_ppf_plot(0.00000001), test_stat_t - 2*std_dev_plot, -4.0) 
        plot_max_t = max(crit_func_ppf_plot(0.99999999), test_stat_t + 2*std_dev_plot, 4.0) 
        if abs(test_stat_t) > 4 and abs(test_stat_t) > plot_max_t * 0.8 : 
            plot_min_t = min(plot_min_t, test_stat_t -1)
            plot_max_t = max(plot_max_t, test_stat_t +1)
        
        x_t_plot = np.linspace(plot_min_t, plot_max_t, 500) 
        y_t_plot = crit_func_pdf_plot(x_t_plot)
        ax_t.plot(x_t_plot, y_t_plot, 'b-', lw=2, label=dist_label_plot)
        
        crit_val_t_upper_plot, crit_val_t_lower_plot = None, None
        if tail_t == "Two-tailed":
            crit_val_t_upper_plot = crit_func_ppf_plot(1 - alpha_t_input / 2)
            crit_val_t_lower_plot = crit_func_ppf_plot(alpha_t_input / 2)
            if crit_val_t_upper_plot is not None and not np.isnan(crit_val_t_upper_plot):
                 x_fill_upper = np.linspace(crit_val_t_upper_plot, plot_max_t, 100)
                 ax_t.fill_between(x_fill_upper, crit_func_pdf_plot(x_fill_upper), color='red', alpha=0.5, label=f'α/2 = {alpha_t_input/2:.8f}')
                 ax_t.axvline(crit_val_t_upper_plot, color='red', linestyle='--', lw=1)
            if crit_val_t_lower_plot is not None and not np.isnan(crit_val_t_lower_plot):
                 x_fill_lower = np.linspace(plot_min_t, crit_val_t_lower_plot, 100)
                 ax_t.fill_between(x_fill_lower, crit_func_pdf_plot(x_fill_lower), color='red', alpha=0.5)
                 ax_t.axvline(crit_val_t_lower_plot, color='red', linestyle='--', lw=1)
        elif tail_t == "One-tailed (right)":
            crit_val_t_upper_plot = crit_func_ppf_plot(1 - alpha_t_input)
            if crit_val_t_upper_plot is not None and not np.isnan(crit_val_t_upper_plot):
                x_fill_upper = np.linspace(crit_val_t_upper_plot, plot_max_t, 100)
                ax_t.fill_between(x_fill_upper, crit_func_pdf_plot(x_fill_upper), color='red', alpha=0.5, label=f'α = {alpha_t_input:.8f}')
                ax_t.axvline(crit_val_t_upper_plot, color='red', linestyle='--', lw=1)
        else: # One-tailed (left)
            crit_val_t_lower_plot = crit_func_ppf_plot(alpha_t_input)
            if crit_val_t_lower_plot is not None and not np.isnan(crit_val_t_lower_plot):
                x_fill_lower = np.linspace(plot_min_t, crit_val_t_lower_plot, 100)
                ax_t.fill_between(x_fill_lower, crit_func_pdf_plot(x_fill_lower), color='red', alpha=0.5, label=f'α = {alpha_t_input:.8f}')
                ax_t.axvline(crit_val_t_lower_plot, color='red', linestyle='--', lw=1)

        ax_t.axvline(test_stat_t, color='green', linestyle='-', lw=2, label=f'Test Stat = {test_stat_t:.3f}')
        ax_t.set_title(f'{dist_label_plot} with Critical Region(s)')
        ax_t.set_xlabel('t-value' if not np.isinf(df_t_calc) else 'z-value')
        ax_t.set_ylabel('Probability Density')
        ax_t.legend()
        ax_t.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_t)

        st.subheader("Critical t-Values (Upper Tail)")
        
        table_df_window = get_dynamic_df_window(all_df_values_t, df_t_selected_display, window_size=5)
        table_alpha_cols = [0.10, 0.05, 0.025, 0.01, 0.005] 

        table_rows = []
        for df_iter_display in table_df_window:
            df_iter_calc = np.inf if df_iter_display == 'z (∞)' else int(df_iter_display)
            row_data = {'df': str(df_iter_display)}
            for alpha_col in table_alpha_cols:
                if np.isinf(df_iter_calc):
                    cv = stats.norm.ppf(1 - alpha_col)
                else:
                    cv = stats.t.ppf(1 - alpha_col, df_iter_calc)
                row_data[f"α = {alpha_col:.3f}"] = format_value_for_display(cv)
            table_rows.append(row_data)
        
        df_t_table = pd.DataFrame(table_rows).set_index('df')

        def style_t_table(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_str = str(df_t_selected_display) 

            if selected_df_str in df_to_style.index: 
                style.loc[selected_df_str, :] = 'background-color: lightblue;'

            target_alpha_for_col_highlight = alpha_t_input 
            if tail_t == "Two-tailed":
                target_alpha_for_col_highlight = alpha_t_input / 2.0
            
            closest_alpha_col_val = min(table_alpha_cols, key=lambda x: abs(x - target_alpha_for_col_highlight))
            highlight_col_name = f"α = {closest_alpha_col_val:.3f}" 

            if highlight_col_name in df_to_style.columns:
                for r_idx in df_to_style.index:
                     current_r_style = style.loc[r_idx, highlight_col_name]
                     style.loc[r_idx, highlight_col_name] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                
                if selected_df_str in df_to_style.index: 
                    current_c_style = style.loc[selected_df_str, highlight_col_name]
                    style.loc[selected_df_str, highlight_col_name] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style

        st.markdown(df_t_table.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                       {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_t_table, axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows upper-tail critical values. Highlighted row for df='{df_t_selected_display}', column for α closest to your test, and specific cell in red.")
        st.markdown("""
        **Table Interpretation Note:**
        * The table displays upper-tail critical values (t<sub>α</sub>).
        * For **One-tailed (right) tests**, use the α column matching your chosen significance level.
        * For **One-tailed (left) tests**, use the α column matching your chosen significance level and take the *negative* of the table value.
        * For **Two-tailed tests**, if your total significance level is α<sub>total</sub>, look up the column for α = α<sub>total</sub>/2. The critical values are ± the table value.
        """)

    with col2: # Summary section
        st.subheader("P-value Calculation Explanation")
        if np.isinf(df_t_calc):
            p_val_func_sf = stats.norm.sf
            p_val_func_cdf = stats.norm.cdf
            dist_name_p_summary = "Z"
        else:
            p_val_func_sf = lambda val: stats.t.sf(val, df_t_calc)
            p_val_func_cdf = lambda val: stats.t.cdf(val, df_t_calc)
            dist_name_p_summary = "T"

        st.markdown(f"""
        The p-value is the probability of observing a test statistic as extreme as, or more extreme than, the calculated statistic ({test_stat_t:.3f}), assuming the null hypothesis is true.
        * For a **two-tailed test**, it's `2 * P({dist_name_p_summary} ≥ |{test_stat_t:.3f}|)`.
        * For a **one-tailed (right) test**, it's `P({dist_name_p_summary} ≥ {test_stat_t:.3f})`.
        * For a **one-tailed (left) test**, it's `P({dist_name_p_summary} ≤ {test_stat_t:.3f})`.
        """)

        st.subheader("Summary")
        p_val_t_one_right_summary = p_val_func_sf(test_stat_t)
        p_val_t_one_left_summary = p_val_func_cdf(test_stat_t)
        p_val_t_two_summary = 2 * p_val_func_sf(abs(test_stat_t))
        p_val_t_two_summary = min(p_val_t_two_summary, 1.0) 

        crit_val_display_summary = "N/A"
        
        if tail_t == "Two-tailed":
            crit_val_display_summary = f"±{format_value_for_display(crit_val_t_upper_plot)}" if crit_val_t_upper_plot is not None else "N/A"
            p_val_calc_summary = p_val_t_two_summary
            decision_crit_summary = abs(test_stat_t) > crit_val_t_upper_plot if crit_val_t_upper_plot is not None and not np.isnan(crit_val_t_upper_plot) else False
            comparison_crit_str_summary = f"|{test_stat_t:.3f}| ({abs(test_stat_t):.3f}) > {format_value_for_display(crit_val_t_upper_plot)}" if decision_crit_summary else f"|{test_stat_t:.3f}| ({abs(test_stat_t):.3f}) ≤ {format_value_for_display(crit_val_t_upper_plot)}"
        elif tail_t == "One-tailed (right)":
            crit_val_display_summary = format_value_for_display(crit_val_t_upper_plot) if crit_val_t_upper_plot is not None else "N/A"
            p_val_calc_summary = p_val_t_one_right_summary
            decision_crit_summary = test_stat_t > crit_val_t_upper_plot if crit_val_t_upper_plot is not None and not np.isnan(crit_val_t_upper_plot) else False
            comparison_crit_str_summary = f"{test_stat_t:.3f} > {format_value_for_display(crit_val_t_upper_plot)}" if decision_crit_summary else f"{test_stat_t:.3f} ≤ {format_value_for_display(crit_val_t_upper_plot)}"
        else: # One-tailed (left)
            crit_val_display_summary = format_value_for_display(crit_val_t_lower_plot) if crit_val_t_lower_plot is not None else "N/A"
            p_val_calc_summary = p_val_t_one_left_summary
            decision_crit_summary = test_stat_t < crit_val_t_lower_plot if crit_val_t_lower_plot is not None and not np.isnan(crit_val_t_lower_plot) else False
            comparison_crit_str_summary = f"{test_stat_t:.3f} < {format_value_for_display(crit_val_t_lower_plot)}" if decision_crit_summary else f"{test_stat_t:.3f} ≥ {format_value_for_display(crit_val_t_lower_plot)}"

        decision_p_alpha_summary = p_val_calc_summary < alpha_t_input
        
        df_report_str_summary = "∞" if np.isinf(df_t_calc) else str(df_t_calc)
        stat_symbol_summary = "z" if np.isinf(df_t_calc) else "t"

        st.markdown(f"""
        1.  **Critical Value ({tail_t})**: {crit_val_display_summary}
            * *Associated p-value (α or α/2 per tail)*: {alpha_t_input:.8f}
        2.  **Calculated Test Statistic**: {test_stat_t:.3f}
            * *Calculated p-value*: {format_value_for_display(p_val_calc_summary, decimals=4)} ({apa_p_value(p_val_calc_summary)})
        3.  **Decision (Critical Value Method)**: The null hypothesis is **{'rejected' if decision_crit_summary else 'not rejected'}**.
            * *Reason*: Because {stat_symbol_summary}(calc) {comparison_crit_str_summary} relative to {stat_symbol_summary}(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_summary else 'not rejected'}**.
            * *Reason*: Because {apa_p_value(p_val_calc_summary)} is {'less than' if decision_p_alpha_summary else 'not less than'} α ({alpha_t_input:.8f}).
        5.  **APA 7 Style Report**:
            *{stat_symbol_summary}*({df_report_str_summary}) = {test_stat_t:.2f}, {apa_p_value(p_val_calc_summary)}. The null hypothesis was {'rejected' if decision_p_alpha_summary else 'not rejected'} at the α = {alpha_t_input:.2f} level.
        """)

# --- Tab 2: z-Distribution ---
def tab_z_distribution():
    st.header("z-Distribution (Standard Normal) Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_z_hyp = st.number_input("Alpha (α) for Hypothesis Test", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_z_hyp_key")
        tail_z_hyp = st.radio("Tail Selection for Hypothesis Test", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_z_hyp_key")
        test_stat_z_hyp = st.number_input("Your Calculated z-statistic (also for Table Lookup)", value=0.0, format="%.3f", key="test_stat_z_hyp_key", min_value=-3.99, max_value=3.99, step=0.01)
        
        z_lookup_val = test_stat_z_hyp 

        st.subheader("Distribution Plot")
        fig_z, ax_z = plt.subplots(figsize=(8,5))
        
        plot_min_z = min(stats.norm.ppf(0.00000001), test_stat_z_hyp - 2, -4.0) 
        plot_max_z = max(stats.norm.ppf(0.99999999), test_stat_z_hyp + 2, 4.0) 
        if abs(test_stat_z_hyp) > 3.5 : 
            plot_min_z = min(plot_min_z, test_stat_z_hyp - 0.5)
            plot_max_z = max(plot_max_z, test_stat_z_hyp + 0.5)

        x_z_plot = np.linspace(plot_min_z, plot_max_z, 500)
        y_z_plot = stats.norm.pdf(x_z_plot)
        ax_z.plot(x_z_plot, y_z_plot, 'b-', lw=2, label='Standard Normal Distribution (z)')

        x_fill_lookup = np.linspace(plot_min_z, z_lookup_val, 100)
        ax_z.fill_between(x_fill_lookup, stats.norm.pdf(x_fill_lookup), color='skyblue', alpha=0.5, label=f'P(Z < {z_lookup_val:.3f})')
        
        crit_val_z_upper_plot, crit_val_z_lower_plot = None, None 
        if tail_z_hyp == "Two-tailed":
            crit_val_z_upper_plot = stats.norm.ppf(1 - alpha_z_hyp / 2)
            crit_val_z_lower_plot = stats.norm.ppf(alpha_z_hyp / 2)
            if crit_val_z_upper_plot is not None and not np.isnan(crit_val_z_upper_plot):
                x_fill_upper = np.linspace(crit_val_z_upper_plot, plot_max_z, 100)
                ax_z.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.3, label=f'Crit. Region α/2')
                ax_z.axvline(crit_val_z_upper_plot, color='red', linestyle='--', lw=1)
            if crit_val_z_lower_plot is not None and not np.isnan(crit_val_z_lower_plot):
                x_fill_lower = np.linspace(plot_min_z, crit_val_z_lower_plot, 100)
                ax_z.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.3)
                ax_z.axvline(crit_val_z_lower_plot, color='red', linestyle='--', lw=1)
        elif tail_z_hyp == "One-tailed (right)":
            crit_val_z_upper_plot = stats.norm.ppf(1 - alpha_z_hyp)
            if crit_val_z_upper_plot is not None and not np.isnan(crit_val_z_upper_plot):
                x_fill_upper = np.linspace(crit_val_z_upper_plot, plot_max_z, 100)
                ax_z.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.3, label=f'Crit. Region α')
                ax_z.axvline(crit_val_z_upper_plot, color='red', linestyle='--', lw=1)
        else: 
            crit_val_z_lower_plot = stats.norm.ppf(alpha_z_hyp)
            if crit_val_z_lower_plot is not None and not np.isnan(crit_val_z_lower_plot):
                x_fill_lower = np.linspace(plot_min_z, crit_val_z_lower_plot, 100)
                ax_z.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.3, label=f'Crit. Region α')
                ax_z.axvline(crit_val_z_lower_plot, color='red', linestyle='--', lw=1)

        ax_z.axvline(test_stat_z_hyp, color='green', linestyle='-', lw=2, label=f'Your z-stat = {test_stat_z_hyp:.3f}')
        ax_z.set_title('Standard Normal Distribution')
        ax_z.set_xlabel('z-value')
        ax_z.set_ylabel('Probability Density')
        ax_z.legend(fontsize='small')
        ax_z.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_z)

        st.subheader("Standard Normal Table: Cumulative P(Z < z)")
        st.markdown("This table shows the area to the left of a given z-score. Your calculated z-statistic is used for highlighting.")
        
        all_z_row_labels = [f"{val:.1f}" for val in np.round(np.arange(-3.4, 3.5, 0.1), 1)]
        z_col_labels_str = [f"{val:.2f}" for val in np.round(np.arange(0.00, 0.10, 0.01), 2)]

        z_target_for_table_row_numeric = round(test_stat_z_hyp, 1) 
        try:
            closest_row_idx = min(range(len(all_z_row_labels)), key=lambda i: abs(float(all_z_row_labels[i]) - z_target_for_table_row_numeric))
        except ValueError: 
            closest_row_idx = len(all_z_row_labels) // 2

        window_size = 5
        start_idx = max(0, closest_row_idx - window_size)
        end_idx = min(len(all_z_row_labels), closest_row_idx + window_size + 1)
        z_table_display_rows_str = all_z_row_labels[start_idx:end_idx]


        table_data_z_lookup = []
        for z_r_str_idx in z_table_display_rows_str:
            z_r_val = float(z_r_str_idx)
            row = { 'z': z_r_str_idx } 
            for z_c_str_idx in z_col_labels_str:
                z_c_val = float(z_c_str_idx)
                current_z_val = round(z_r_val + z_c_val, 2)
                prob = stats.norm.cdf(current_z_val)
                row[z_c_str_idx] = format_value_for_display(prob, decimals=4)
            table_data_z_lookup.append(row)
        
        df_z_lookup_table = pd.DataFrame(table_data_z_lookup).set_index('z')

        def style_z_lookup_table(df_to_style):
            data = df_to_style 
            style_df = pd.DataFrame('', index=data.index, columns=data.columns)
            
            try:
                z_target = test_stat_z_hyp 
                
                z_target_base_numeric = round(z_target,1) 
                actual_row_labels_float = [float(label) for label in data.index]
                closest_row_float_val = min(actual_row_labels_float, key=lambda x_val: abs(x_val - z_target_base_numeric))
                highlight_row_label = f"{closest_row_float_val:.1f}"

                z_target_second_decimal_part = round(abs(z_target - closest_row_float_val), 2) 
                
                actual_col_labels_float = [float(col_str) for col_str in data.columns]
                closest_col_float_val = min(actual_col_labels_float, key=lambda x_val: abs(x_val - z_target_second_decimal_part))
                highlight_col_label = f"{closest_col_float_val:.2f}"


                if highlight_row_label in style_df.index:
                    for col_name_iter in style_df.columns: 
                        style_df.loc[highlight_row_label, col_name_iter] = 'background-color: lightblue;'
                
                if highlight_col_label in style_df.columns:
                    for r_idx_iter in style_df.index: 
                        current_style = style_df.loc[r_idx_iter, highlight_col_label]
                        style_df.loc[r_idx_iter, highlight_col_label] = (current_style + ';' if current_style and not current_style.endswith(';') else current_style) + 'background-color: lightgreen;'
                
                if highlight_row_label in style_df.index and highlight_col_label in style_df.columns:
                    current_cell_style = style_df.loc[highlight_row_label, highlight_col_label]
                    style_df.loc[highlight_row_label, highlight_col_label] = (current_cell_style + ';' if current_cell_style and not current_cell_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            
            except Exception: 
                pass 
            return style_df
        
        st.markdown(df_z_lookup_table.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                               {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_z_lookup_table, axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows P(Z < z). Highlighted cell, row, and column are closest to your calculated z-statistic of {test_stat_z_hyp:.3f}.")


    with col2: # Summary for Z-distribution hypothesis test
        st.subheader("Hypothesis Test Summary")
        st.markdown(f"""
        Based on your inputs for hypothesis testing (α = {alpha_z_hyp:.8f}, {tail_z_hyp}):
        """)
        p_val_z_one_right_summary = stats.norm.sf(test_stat_z_hyp)
        p_val_z_one_left_summary = stats.norm.cdf(test_stat_z_hyp)
        p_val_z_two_summary = 2 * stats.norm.sf(abs(test_stat_z_hyp))
        p_val_z_two_summary = min(p_val_z_two_summary, 1.0)

        crit_val_display_z = "N/A"
        
        if tail_z_hyp == "Two-tailed":
            crit_val_display_z = f"±{format_value_for_display(crit_val_z_upper_plot)}" if crit_val_z_upper_plot is not None else "N/A"
            p_val_calc_z_summary = p_val_z_two_summary
            decision_crit_z_summary = abs(test_stat_z_hyp) > crit_val_z_upper_plot if crit_val_z_upper_plot is not None and not np.isnan(crit_val_z_upper_plot) else False
            comparison_crit_str_z = f"|{test_stat_z_hyp:.3f}| ({abs(test_stat_z_hyp):.3f}) > {format_value_for_display(crit_val_z_upper_plot)}" if decision_crit_z_summary else f"|{test_stat_z_hyp:.3f}| ({abs(test_stat_z_hyp):.3f}) ≤ {format_value_for_display(crit_val_z_upper_plot)}"
        elif tail_z_hyp == "One-tailed (right)":
            crit_val_display_z = format_value_for_display(crit_val_z_upper_plot) if crit_val_z_upper_plot is not None else "N/A"
            p_val_calc_z_summary = p_val_z_one_right_summary
            decision_crit_z_summary = test_stat_z_hyp > crit_val_z_upper_plot if crit_val_z_upper_plot is not None and not np.isnan(crit_val_z_upper_plot) else False
            comparison_crit_str_z = f"{test_stat_z_hyp:.3f} > {format_value_for_display(crit_val_z_upper_plot)}" if decision_crit_z_summary else f"{test_stat_z_hyp:.3f} ≤ {format_value_for_display(crit_val_z_upper_plot)}"
        else: # One-tailed (left)
            crit_val_display_z = format_value_for_display(crit_val_z_lower_plot) if crit_val_z_lower_plot is not None else "N/A"
            p_val_calc_z_summary = p_val_z_one_left_summary
            decision_crit_z_summary = test_stat_z_hyp < crit_val_z_lower_plot if crit_val_z_lower_plot is not None and not np.isnan(crit_val_z_lower_plot) else False
            comparison_crit_str_z = f"{test_stat_z_hyp:.3f} < {format_value_for_display(crit_val_z_lower_plot)}" if decision_crit_z_summary else f"{test_stat_z_hyp:.3f} ≥ {format_value_for_display(crit_val_z_lower_plot)}"

        decision_p_alpha_z_summary = p_val_calc_z_summary < alpha_z_hyp
        
        st.markdown(f"""
        1.  **Critical Value ({tail_z_hyp})**: {crit_val_display_z}
            * *Associated p-value (α or α/2 per tail)*: {alpha_z_hyp:.8f}
        2.  **Your Calculated Test Statistic**: {test_stat_z_hyp:.3f}
            * *Calculated p-value*: {format_value_for_display(p_val_calc_z_summary, decimals=4)} ({apa_p_value(p_val_calc_z_summary)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_z_summary else 'not rejected'}**.
            * *Reason*: z(calc) {comparison_crit_str_z} relative to z(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_z_summary else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_z_summary)} is {'less than' if decision_p_alpha_z_summary else 'not less than'} α ({alpha_z_hyp:.8f}).
        5.  **APA 7 Style Report**:
            *z* = {test_stat_z_hyp:.2f}, {apa_p_value(p_val_calc_z_summary)}. The null hypothesis was {'rejected' if decision_p_alpha_z_summary else 'not rejected'} at α = {alpha_z_hyp:.2f}.
        """)


# --- Tab 3: F-distribution ---
def tab_f_distribution():
    st.header("F-Distribution Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_f_input = st.number_input("Alpha (α) for Table/Plot", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_f_input_tab3")
        
        all_df1_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 24, 30, 40, 50, 60, 80, 100, 120, 1000]
        all_df2_options = list(range(1,21)) + [22, 24, 26, 28, 30, 35, 40, 45, 50, 60, 80, 100, 120, 1000] 

        df1_f_selected = st.selectbox("Numerator df (df₁) for Plot & Summary", options=all_df1_options, index=all_df1_options.index(3), key="df1_f_selectbox_tab3") 
        df2_f_selected = st.selectbox("Denominator df (df₂) for Plot & Summary", options=all_df2_options, index=all_df2_options.index(20), key="df2_f_selectbox_tab3") 
        
        tail_f = st.radio("Tail Selection (for plot & summary)", ("One-tailed (right)", "Two-tailed (for variance test)"), key="tail_f_radio_tab3")
        test_stat_f = st.number_input("Calculated F-statistic", value=1.0, format="%.3f", min_value=0.001, key="test_stat_f_input_tab3")

        st.subheader("Distribution Plot")
        fig_f, ax_f = plt.subplots(figsize=(8,5))
        
        plot_min_f = 0.001
        plot_max_f = 5.0 
        try:
            plot_max_f = max(stats.f.ppf(0.999, df1_f_selected, df2_f_selected), test_stat_f * 1.5, 5.0)
            if test_stat_f > stats.f.ppf(0.999, df1_f_selected, df2_f_selected) * 1.2 : 
                plot_max_f = test_stat_f * 1.2
        except Exception: pass

        x_f_plot = np.linspace(plot_min_f, plot_max_f, 500)
        y_f_plot = stats.f.pdf(x_f_plot, df1_f_selected, df2_f_selected)
        ax_f.plot(x_f_plot, y_f_plot, 'b-', lw=2, label=f'F-dist (df₁={df1_f_selected}, df₂={df2_f_selected})')

        crit_val_f_upper_plot, crit_val_f_lower_plot = None, None
        alpha_for_plot = alpha_f_input 
        if tail_f == "One-tailed (right)":
            crit_val_f_upper_plot = stats.f.ppf(1 - alpha_for_plot, df1_f_selected, df2_f_selected)
            if crit_val_f_upper_plot is not None and not np.isnan(crit_val_f_upper_plot):
                x_fill_upper = np.linspace(crit_val_f_upper_plot, plot_max_f, 100)
                ax_f.fill_between(x_fill_upper, stats.f.pdf(x_fill_upper, df1_f_selected, df2_f_selected), color='red', alpha=0.5, label=f'α = {alpha_for_plot:.8f}')
                ax_f.axvline(crit_val_f_upper_plot, color='red', linestyle='--', lw=1)
        else: 
            crit_val_f_upper_plot = stats.f.ppf(1 - alpha_for_plot / 2, df1_f_selected, df2_f_selected)
            crit_val_f_lower_plot = stats.f.ppf(alpha_for_plot / 2, df1_f_selected, df2_f_selected)
            if crit_val_f_upper_plot is not None and not np.isnan(crit_val_f_upper_plot):
                x_fill_upper = np.linspace(crit_val_f_upper_plot, plot_max_f, 100)
                ax_f.fill_between(x_fill_upper, stats.f.pdf(x_fill_upper, df1_f_selected, df2_f_selected), color='red', alpha=0.5, label=f'α/2 = {alpha_for_plot/2:.8f}')
                ax_f.axvline(crit_val_f_upper_plot, color='red', linestyle='--', lw=1)
            if crit_val_f_lower_plot is not None and not np.isnan(crit_val_f_lower_plot):
                x_fill_lower = np.linspace(plot_min_f, crit_val_f_lower_plot, 100)
                ax_f.fill_between(x_fill_lower, stats.f.pdf(x_fill_lower, df1_f_selected, df2_f_selected), color='red', alpha=0.5)
                ax_f.axvline(crit_val_f_lower_plot, color='red', linestyle='--', lw=1)

        ax_f.axvline(test_stat_f, color='green', linestyle='-', lw=2, label=f'Test Stat = {test_stat_f:.3f}')
        ax_f.set_title(f'F-Distribution (df₁={df1_f_selected}, df₂={df2_f_selected}) with Critical Region(s)')
        ax_f.set_xlabel('F-value')
        ax_f.set_ylabel('Probability Density')
        ax_f.legend()
        ax_f.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_f)

        st.subheader(f"Critical F-Values for α = {alpha_f_input:.8f} (Upper Tail)")
        table_df1_display_cols = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 60, 120] 
        table_df2_window = get_dynamic_df_window(all_df2_options, df2_f_selected, window_size=5)


        f_table_data = []
        for df2_val_iter in table_df2_window: 
            df2_val_calc = int(df2_val_iter) 
            row = {'df₂': str(df2_val_iter)} 
            for df1_val_iter in table_df1_display_cols: 
                cv = stats.f.ppf(1 - alpha_f_input, df1_val_iter, df2_val_calc)
                row[f"df₁={df1_val_iter}"] = format_value_for_display(cv)
            f_table_data.append(row)
        
        df_f_table = pd.DataFrame(f_table_data).set_index('df₂')

        def style_f_table(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df2_str = str(df2_f_selected) 
            
            if selected_df2_str in df_to_style.index:
                style.loc[selected_df2_str, :] = 'background-color: lightblue;'
            
            closest_df1_col_val = min(table_df1_display_cols, key=lambda x: abs(x - df1_f_selected))
            highlight_col_name_f = f"df₁={closest_df1_col_val}"

            if highlight_col_name_f in df_to_style.columns:
                for r_idx in df_to_style.index:
                    current_r_style = style.loc[r_idx, highlight_col_name_f]
                    style.loc[r_idx, highlight_col_name_f] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                
                if selected_df2_str in df_to_style.index : 
                    current_c_style = style.loc[selected_df2_str, highlight_col_name_f]
                    style.loc[selected_df2_str, highlight_col_name_f] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style

        st.markdown(df_f_table.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                       {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_f_table, axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows F-critical values for user-selected α={alpha_f_input:.8f} (upper tail). Highlighted for df₁ closest to {df1_f_selected} and df₂ closest to {df2_f_selected}.")
        st.markdown("""
        **Table Interpretation Note:**
        * This table shows upper-tail critical values F<sub>α, df₁, df₂</sub> for the selected α.
        * For **ANOVA (typically one-tailed right)**, use this table directly with your chosen α.
        * For **Two-tailed variance tests** (H₀: σ₁²=σ₂² vs H₁: σ₁²≠σ₂²), you need two critical values:
            * Upper: F<sub>α/2, df₁, df₂</sub> (Look up using α/2 with this table).
            * Lower: F<sub>1-α/2, df₁, df₂</sub> = 1 / F<sub>α/2, df₂, df₁</sub> (requires swapping df and taking reciprocal of value from table looked up with α/2).
        """)

    with col2: # Summary for F-distribution
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value is the probability of an F-statistic as extreme as, or more extreme than, {test_stat_f:.3f}.
        * **One-tailed (right)**: `P(F ≥ {test_stat_f:.3f})` (i.e., `stats.f.sf(test_stat_f, df1_f_selected, df2_f_selected)`)
        * **Two-tailed (for variance test)**: `2 * min(P(F ≤ F_calc), P(F ≥ F_calc))` (i.e., `2 * min(stats.f.cdf(test_stat_f, df1_f_selected, df2_f_selected), stats.f.sf(test_stat_f, df1_f_selected, df2_f_selected))`)
        """)

        st.subheader("Summary")
        p_val_f_one_right_summary = stats.f.sf(test_stat_f, df1_f_selected, df2_f_selected)
        cdf_f_summary = stats.f.cdf(test_stat_f, df1_f_selected, df2_f_selected)
        sf_f_summary = stats.f.sf(test_stat_f, df1_f_selected, df2_f_selected)
        p_val_f_two_summary = 2 * min(cdf_f_summary, sf_f_summary)
        p_val_f_two_summary = min(p_val_f_two_summary, 1.0)

        crit_val_f_display_summary = "N/A"
        
        if tail_f == "One-tailed (right)":
            crit_val_f_display_summary = format_value_for_display(crit_val_f_upper_plot) if crit_val_f_upper_plot is not None else "N/A"
            p_val_calc_f_summary = p_val_f_one_right_summary
            decision_crit_f_summary = test_stat_f > crit_val_f_upper_plot if crit_val_f_upper_plot is not None and not np.isnan(crit_val_f_upper_plot) else False
            comparison_crit_str_f = f"{test_stat_f:.3f} > {format_value_for_display(crit_val_f_upper_plot)}" if decision_crit_f_summary else f"{test_stat_f:.3f} ≤ {format_value_for_display(crit_val_f_upper_plot)}"
        else: # Two-tailed
            crit_val_f_display_summary = f"Lower: {format_value_for_display(crit_val_f_lower_plot)}, Upper: {format_value_for_display(crit_val_f_upper_plot)}" \
                                         if crit_val_f_lower_plot is not None and crit_val_f_upper_plot is not None else "N/A"
            p_val_calc_f_summary = p_val_f_two_summary
            decision_crit_f_summary = (test_stat_f > crit_val_f_upper_plot if crit_val_f_upper_plot is not None and not np.isnan(crit_val_f_upper_plot) else False) or \
                                     (test_stat_f < crit_val_f_lower_plot if crit_val_f_lower_plot is not None and not np.isnan(crit_val_f_lower_plot) else False)
            comparison_crit_str_f = f"{test_stat_f:.3f} > {format_value_for_display(crit_val_f_upper_plot)} or {test_stat_f:.3f} < {format_value_for_display(crit_val_f_lower_plot)}" if decision_crit_f_summary else f"{format_value_for_display(crit_val_f_lower_plot)} ≤ {test_stat_f:.3f} ≤ {format_value_for_display(crit_val_f_upper_plot)}"
        
        decision_p_alpha_f_summary = p_val_calc_f_summary < alpha_f_input 
        
        st.markdown(f"""
        1.  **Critical Value(s) ({tail_f}) for α={alpha_f_input:.8f}**: {crit_val_f_display_summary}
            * *Associated p-value (α or α/2 per tail)*: {alpha_f_input:.8f}
        2.  **Calculated Test Statistic**: {test_stat_f:.3f}
            * *Calculated p-value*: {format_value_for_display(p_val_calc_f_summary, decimals=4)} ({apa_p_value(p_val_calc_f_summary)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_f_summary else 'not rejected'}**.
            * *Reason*: F(calc) {comparison_crit_str_f} relative to F(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_f_summary else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_f_summary)} is {'less than' if decision_p_alpha_f_summary else 'not less than'} α ({alpha_f_input:.8f}).
        5.  **APA 7 Style Report**:
            *F*({df1_f_selected}, {df2_f_selected}) = {test_stat_f:.2f}, {apa_p_value(p_val_calc_f_summary)}. The null hypothesis was {'rejected' if decision_p_alpha_f_summary else 'not rejected'} at α = {alpha_f_input:.2f}.
        """)

# --- Tab 4: Chi-square Distribution ---
def tab_chi_square_distribution():
    st.header("Chi-square (χ²) Distribution Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_chi2_input = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_chi2_input_tab4")
        all_df_chi2_options = list(range(1, 31)) + [35, 40, 45, 50, 60, 70, 80, 90, 100]
        df_chi2_selected = st.selectbox("Degrees of Freedom (df)", options=all_df_chi2_options, index=all_df_chi2_options.index(5), key="df_chi2_selectbox_tab4") 
        
        tail_chi2 = st.radio("Tail Selection", ("One-tailed (right)", "Two-tailed (e.g. for variance)"), key="tail_chi2_radio_tab4")
        test_stat_chi2 = st.number_input("Calculated χ²-statistic", value=float(df_chi2_selected), format="%.3f", min_value=0.001, key="test_stat_chi2_input_tab4")

        st.subheader("Distribution Plot")
        fig_chi2, ax_chi2 = plt.subplots(figsize=(8,5))
        
        plot_min_chi2 = 0.001
        plot_max_chi2 = 10.0 
        try:
            plot_max_chi2 = max(stats.chi2.ppf(0.999, df_chi2_selected), test_stat_chi2 * 1.5, 10.0)
            if test_stat_chi2 > stats.chi2.ppf(0.999, df_chi2_selected) * 1.2:
                plot_max_chi2 = test_stat_chi2 * 1.2
        except Exception: pass

        x_chi2_plot = np.linspace(plot_min_chi2, plot_max_chi2, 500) 
        y_chi2_plot = stats.chi2.pdf(x_chi2_plot, df_chi2_selected)
        ax_chi2.plot(x_chi2_plot, y_chi2_plot, 'b-', lw=2, label=f'χ²-distribution (df={df_chi2_selected})')

        crit_val_chi2_upper_plot, crit_val_chi2_lower_plot = None, None
        if tail_chi2 == "One-tailed (right)":
            crit_val_chi2_upper_plot = stats.chi2.ppf(1 - alpha_chi2_input, df_chi2_selected)
            if crit_val_chi2_upper_plot is not None and not np.isnan(crit_val_chi2_upper_plot):
                x_fill_upper = np.linspace(crit_val_chi2_upper_plot, plot_max_chi2, 100)
                ax_chi2.fill_between(x_fill_upper, stats.chi2.pdf(x_fill_upper, df_chi2_selected), color='red', alpha=0.5, label=f'α = {alpha_chi2_input:.8f}')
                ax_chi2.axvline(crit_val_chi2_upper_plot, color='red', linestyle='--', lw=1)
        else: 
            crit_val_chi2_upper_plot = stats.chi2.ppf(1 - alpha_chi2_input / 2, df_chi2_selected)
            crit_val_chi2_lower_plot = stats.chi2.ppf(alpha_chi2_input / 2, df_chi2_selected)
            if crit_val_chi2_upper_plot is not None and not np.isnan(crit_val_chi2_upper_plot):
                x_fill_upper_chi2 = np.linspace(crit_val_chi2_upper_plot, plot_max_chi2, 100)
                ax_chi2.fill_between(x_fill_upper_chi2, stats.chi2.pdf(x_fill_upper_chi2, df_chi2_selected), color='red', alpha=0.5, label=f'α/2 = {alpha_chi2_input/2:.8f}')
                ax_chi2.axvline(crit_val_chi2_upper_plot, color='red', linestyle='--', lw=1)
            if crit_val_chi2_lower_plot is not None and not np.isnan(crit_val_chi2_lower_plot):
                x_fill_lower_chi2 = np.linspace(plot_min_chi2, crit_val_chi2_lower_plot, 100)
                ax_chi2.fill_between(x_fill_lower_chi2, stats.chi2.pdf(x_fill_lower_chi2, df_chi2_selected), color='red', alpha=0.5)
                ax_chi2.axvline(crit_val_chi2_lower_plot, color='red', linestyle='--', lw=1)

        ax_chi2.axvline(test_stat_chi2, color='green', linestyle='-', lw=2, label=f'Test Stat = {test_stat_chi2:.3f}')
        ax_chi2.set_title(f'χ²-Distribution (df={df_chi2_selected}) with Critical Region(s)')
        ax_chi2.set_xlabel('χ²-value')
        ax_chi2.set_ylabel('Probability Density')
        ax_chi2.legend(); ax_chi2.grid(True); st.pyplot(fig_chi2)

        st.subheader("Critical χ²-Values (Upper Tail)")
        table_df_window_chi2 = get_dynamic_df_window(all_df_chi2_options, df_chi2_selected, window_size=5)
        table_alpha_cols_chi2 = [0.10, 0.05, 0.025, 0.01, 0.005] 
        
        chi2_table_rows = []
        for df_iter_val in table_df_window_chi2:
            df_iter_calc = int(df_iter_val)
            row_data = {'df': str(df_iter_val)}
            for alpha_col in table_alpha_cols_chi2:
                cv = stats.chi2.ppf(1 - alpha_col, df_iter_calc) 
                row_data[f"α = {alpha_col:.3f}"] = format_value_for_display(cv)
            chi2_table_rows.append(row_data)
        
        df_chi2_table = pd.DataFrame(chi2_table_rows).set_index('df')

        def style_chi2_table(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_str = str(df_chi2_selected)

            if selected_df_str in df_to_style.index:
                style.loc[selected_df_str, :] = 'background-color: lightblue;'
            
            target_alpha_for_col_highlight = alpha_chi2_input
            if tail_chi2 == "Two-tailed (e.g. for variance)":
                 target_alpha_for_col_highlight = alpha_chi2_input / 2.0 

            closest_alpha_col_val = min(table_alpha_cols_chi2, key=lambda x: abs(x - target_alpha_for_col_highlight))
            highlight_col_name = f"α = {closest_alpha_col_val:.3f}"

            if highlight_col_name in df_to_style.columns:
                for r_idx in df_to_style.index:
                     current_r_style = style.loc[r_idx, highlight_col_name]
                     style.loc[r_idx, highlight_col_name] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                if selected_df_str in df_to_style.index:
                    current_c_style = style.loc[selected_df_str, highlight_col_name]
                    style.loc[selected_df_str, highlight_col_name] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style
        
        st.markdown(df_chi2_table.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                       {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_chi2_table, axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows upper-tail critical χ²-values. Highlighted for df={df_chi2_selected} and α closest to your test.")
        st.markdown("""
        **Table Interpretation Note:**
        * The table displays upper-tail critical values (χ²<sub>α</sub>).
        * For **One-tailed (right) tests** (e.g., goodness-of-fit, independence), use the α column matching your chosen significance level.
        * For **Two-tailed tests on variance**, if your total significance level is α<sub>total</sub>:
            * Upper critical value: Look up column for α = α<sub>total</sub>/2.
            * Lower critical value: Use `stats.chi2.ppf(α_total/2, df)` (not directly in this table's main columns).
        """)


    with col2: # Summary for Chi-square
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value is the probability of a χ²-statistic as extreme as, or more extreme than, {test_stat_chi2:.3f}.
        * **One-tailed (right)**: `P(χ² ≥ {test_stat_chi2:.3f})` (i.e., `stats.chi2.sf(test_stat_chi2, df_chi2_selected)`)
        * **Two-tailed**: `2 * min(P(χ² ≤ {test_stat_chi2:.3f}), P(χ² ≥ {test_stat_chi2:.3f}))` (i.e., `2 * min(stats.chi2.cdf(test_stat_chi2, df_chi2_selected), stats.chi2.sf(test_stat_chi2, df_chi2_selected))`)
        """)

        st.subheader("Summary")
        p_val_chi2_one_right_summary = stats.chi2.sf(test_stat_chi2, df_chi2_selected)
        cdf_chi2_summary = stats.chi2.cdf(test_stat_chi2, df_chi2_selected)
        sf_chi2_summary = stats.chi2.sf(test_stat_chi2, df_chi2_selected)
        p_val_chi2_two_summary = 2 * min(cdf_chi2_summary, sf_chi2_summary)
        p_val_chi2_two_summary = min(p_val_chi2_two_summary, 1.0)

        crit_val_chi2_display_summary = "N/A"
        
        if tail_chi2 == "One-tailed (right)":
            crit_val_chi2_display_summary = format_value_for_display(crit_val_chi2_upper_plot) if crit_val_chi2_upper_plot is not None else "N/A"
            p_val_calc_chi2_summary = p_val_chi2_one_right_summary
            decision_crit_chi2_summary = test_stat_chi2 > crit_val_chi2_upper_plot if crit_val_chi2_upper_plot is not None and not np.isnan(crit_val_chi2_upper_plot) else False
            comparison_crit_str_chi2 = f"{test_stat_chi2:.3f} > {format_value_for_display(crit_val_chi2_upper_plot)}" if decision_crit_chi2_summary else f"{test_stat_chi2:.3f} ≤ {format_value_for_display(crit_val_chi2_upper_plot)}"
        else: # Two-tailed
            crit_val_chi2_display_summary = f"Lower: {format_value_for_display(crit_val_chi2_lower_plot)}, Upper: {format_value_for_display(crit_val_chi2_upper_plot)}" \
                                            if crit_val_chi2_lower_plot is not None and crit_val_chi2_upper_plot is not None else "N/A"
            p_val_calc_chi2_summary = p_val_chi2_two_summary
            decision_crit_chi2_summary = (test_stat_chi2 > crit_val_chi2_upper_plot if crit_val_chi2_upper_plot is not None and not np.isnan(crit_val_chi2_upper_plot) else False) or \
                                         (test_stat_chi2 < crit_val_chi2_lower_plot if crit_val_chi2_lower_plot is not None and not np.isnan(crit_val_chi2_lower_plot) else False)
            comparison_crit_str_chi2 = f"{test_stat_chi2:.3f} > {format_value_for_display(crit_val_chi2_upper_plot)} or {test_stat_chi2:.3f} < {format_value_for_display(crit_val_chi2_lower_plot)}" if decision_crit_chi2_summary else f"{format_value_for_display(crit_val_chi2_lower_plot)} ≤ {test_stat_chi2:.3f} ≤ {format_value_for_display(crit_val_chi2_upper_plot)}"

        decision_p_alpha_chi2_summary = p_val_calc_chi2_summary < alpha_chi2_input
        
        st.markdown(f"""
        1.  **Critical Value(s) ({tail_chi2})**: {crit_val_chi2_display_summary}
            * *Associated p-value (α or α/2 per tail)*: {alpha_chi2_input:.8f} 
        2.  **Calculated Test Statistic**: {test_stat_chi2:.3f}
            * *Calculated p-value*: {format_value_for_display(p_val_calc_chi2_summary, decimals=4)} ({apa_p_value(p_val_calc_chi2_summary)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_chi2_summary else 'not rejected'}**.
            * *Reason*: χ²(calc) {comparison_crit_str_chi2} relative to χ²(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_chi2_summary else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_chi2_summary)} is {'less than' if decision_p_alpha_chi2_summary else 'not less than'} α ({alpha_chi2_input:.8f}).
        5.  **APA 7 Style Report**:
            χ²({df_chi2_selected}) = {test_stat_chi2:.2f}, {apa_p_value(p_val_calc_chi2_summary)}. The null hypothesis was {'rejected' if decision_p_alpha_chi2_summary else 'not rejected'} at α = {alpha_chi2_input:.2f}.
        """)


# --- Tab 5: Mann-Whitney U Test ---
def tab_mann_whitney_u():
    st.header("Mann-Whitney U Test (Normal Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_mw = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_mw_input")
        n1_mw = st.number_input("Sample Size Group 1 (n1)", 1, 1000, 10, 1, key="n1_mw_input") 
        n2_mw = st.number_input("Sample Size Group 2 (n2)", 1, 1000, 12, 1, key="n2_mw_input") 
        tail_mw = st.radio("Tail Selection", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_mw_radio")
        u_stat_mw = st.number_input("Calculated U-statistic", value=float(n1_mw*n2_mw/2), format="%.1f", min_value=0.0, max_value=float(n1_mw*n2_mw), key="u_stat_mw_input")
        
        st.info("This tab uses the Normal Approximation for the Mann-Whitney U test. For small samples (e.g., n1 or n2 < 10), this is an approximation; exact methods/tables are preferred for definitive conclusions.")

        mu_u = (n1_mw * n2_mw) / 2
        sigma_u_sq = (n1_mw * n2_mw * (n1_mw + n2_mw + 1)) / 12
        sigma_u = np.sqrt(sigma_u_sq) if sigma_u_sq > 0 else 0
        
        z_calc_mw = 0.0
        if sigma_u > 0:
            if u_stat_mw < mu_u: z_calc_mw = (u_stat_mw + 0.5 - mu_u) / sigma_u
            elif u_stat_mw > mu_u: z_calc_mw = (u_stat_mw - 0.5 - mu_u) / sigma_u
        elif n1_mw > 0 and n2_mw > 0 : 
            st.warning("Standard deviation (σ_U) for normal approximation is zero. Check sample sizes. z_calc set to 0.")


        st.markdown(f"**Normal Approx. Parameters:** μ<sub>U</sub> = {mu_u:.2f}, σ<sub>U</sub> = {sigma_u:.2f}")
        st.markdown(f"**z-statistic (from U, for approx.):** {z_calc_mw:.3f}")

        st.subheader("Distribution Plot (Normal Approximation)")
        fig_mw, ax_mw = plt.subplots(figsize=(8,5))
        plot_min_z_mw = min(stats.norm.ppf(0.00000001), z_calc_mw - 2, -4.0)
        plot_max_z_mw = max(stats.norm.ppf(0.99999999), z_calc_mw + 2, 4.0)
        if abs(z_calc_mw) > 4 and abs(z_calc_mw) > plot_max_z_mw * 0.8:
            plot_min_z_mw = min(plot_min_z_mw, z_calc_mw -1); plot_max_z_mw = max(plot_max_z_mw, z_calc_mw +1)
        
        x_norm_mw = np.linspace(plot_min_z_mw, plot_max_z_mw, 500)
        y_norm_mw = stats.norm.pdf(x_norm_mw)
        ax_mw.plot(x_norm_mw, y_norm_mw, 'b-', lw=2, label='Standard Normal Distribution (Approx.)')
        
        crit_z_upper_mw_plot, crit_z_lower_mw_plot = None, None
        if tail_mw == "Two-tailed": crit_z_upper_mw_plot = stats.norm.ppf(1 - alpha_mw / 2); crit_z_lower_mw_plot = stats.norm.ppf(alpha_mw / 2)
        elif tail_mw == "One-tailed (right)": crit_z_upper_mw_plot = stats.norm.ppf(1 - alpha_mw)
        else: crit_z_lower_mw_plot = stats.norm.ppf(alpha_mw)

        if crit_z_upper_mw_plot is not None and not np.isnan(crit_z_upper_mw_plot):
            x_fill_upper = np.linspace(crit_z_upper_mw_plot, plot_max_z_mw, 100)
            ax_mw.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, label=f'Approx. Crit. Region')
            ax_mw.axvline(crit_z_upper_mw_plot, color='red', linestyle='--', lw=1)
        if crit_z_lower_mw_plot is not None and not np.isnan(crit_z_lower_mw_plot):
            x_fill_lower = np.linspace(plot_min_z_mw, crit_z_lower_mw_plot, 100)
            ax_mw.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.5, label=f'Approx. Crit. Region' if crit_z_upper_mw_plot is None else None) 
            ax_mw.axvline(crit_z_lower_mw_plot, color='red', linestyle='--', lw=1)

        ax_mw.axvline(z_calc_mw, color='green', linestyle='-', lw=2, label=f'Approx. z_calc = {z_calc_mw:.3f}')
        ax_mw.set_title('Normal Approximation for Mann-Whitney U'); ax_mw.legend(); ax_mw.grid(True); st.pyplot(fig_mw)

        st.subheader("Standard Normal Table: Cumulative P(Z < z)")
        st.markdown("This table shows the area to the left of a given z-score. The z-critical value (derived from your alpha and tail selection) is used for highlighting.")
        
        # Determine z-critical for table centering
        if tail_mw == "Two-tailed":
            z_crit_for_table_mw = crit_z_upper_mw_plot if crit_z_upper_mw_plot is not None else 0.0
        elif tail_mw == "One-tailed (right)":
            z_crit_for_table_mw = crit_z_upper_mw_plot if crit_z_upper_mw_plot is not None else 0.0
        else: # One-tailed (left)
            z_crit_for_table_mw = crit_z_lower_mw_plot if crit_z_lower_mw_plot is not None else 0.0
        if z_crit_for_table_mw is None or np.isnan(z_crit_for_table_mw): z_crit_for_table_mw = 0.0


        all_z_row_labels_mw = [f"{val:.1f}" for val in np.round(np.arange(-3.4, 3.5, 0.1), 1)]
        z_col_labels_str_mw = [f"{val:.2f}" for val in np.round(np.arange(0.00, 0.10, 0.01), 2)]
        
        z_target_for_table_row_numeric_mw = round(z_crit_for_table_mw, 1) 

        try:
            closest_row_idx_mw = min(range(len(all_z_row_labels_mw)), key=lambda i: abs(float(all_z_row_labels_mw[i]) - z_target_for_table_row_numeric_mw))
        except ValueError: 
            closest_row_idx_mw = len(all_z_row_labels_mw) // 2

        window_size_z_mw = 5
        start_idx_z_mw = max(0, closest_row_idx_mw - window_size_z_mw)
        end_idx_z_mw = min(len(all_z_row_labels_mw), closest_row_idx_mw + window_size_z_mw + 1)
        z_table_display_rows_str_mw = all_z_row_labels_mw[start_idx_z_mw:end_idx_z_mw]

        table_data_z_lookup_mw = []
        for z_r_str_idx in z_table_display_rows_str_mw:
            z_r_val = float(z_r_str_idx)
            row = { 'z': z_r_str_idx } 
            for z_c_str_idx in z_col_labels_str_mw:
                z_c_val = float(z_c_str_idx)
                current_z_val = round(z_r_val + z_c_val, 2)
                prob = stats.norm.cdf(current_z_val)
                row[z_c_str_idx] = format_value_for_display(prob, decimals=4)
            table_data_z_lookup_mw.append(row)
        
        df_z_lookup_table_mw = pd.DataFrame(table_data_z_lookup_mw).set_index('z')

        def style_z_lookup_table_mw(df_to_style): 
            data = df_to_style 
            style_df = pd.DataFrame('', index=data.index, columns=data.columns)
            z_crit_val_to_highlight = z_crit_for_table_mw
            try:
                z_target_base_numeric = round(z_crit_val_to_highlight,1) 
                actual_row_labels_float = [float(label) for label in data.index]
                closest_row_float_val = min(actual_row_labels_float, key=lambda x_val: abs(x_val - z_target_base_numeric))
                highlight_row_label = f"{closest_row_float_val:.1f}"

                z_target_second_decimal_part = round(abs(z_crit_val_to_highlight - closest_row_float_val), 2) 
                actual_col_labels_float = [float(col_str) for col_str in data.columns]
                closest_col_float_val = min(actual_col_labels_float, key=lambda x_val: abs(x_val - z_target_second_decimal_part))
                highlight_col_label = f"{closest_col_float_val:.2f}"

                if highlight_row_label in style_df.index:
                    for col_name_iter in style_df.columns: 
                        style_df.loc[highlight_row_label, col_name_iter] = 'background-color: lightblue;'
                if highlight_col_label in style_df.columns:
                    for r_idx_iter in style_df.index: 
                        current_style = style_df.loc[r_idx_iter, highlight_col_label]
                        style_df.loc[r_idx_iter, highlight_col_label] = (current_style + ';' if current_style and not current_style.endswith(';') else current_style) + 'background-color: lightgreen;'
                if highlight_row_label in style_df.index and highlight_col_label in style_df.columns:
                    current_cell_style = style_df.loc[highlight_row_label, highlight_col_label]
                    style_df.loc[highlight_row_label, highlight_col_label] = (current_cell_style + ';' if current_cell_style and not current_cell_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            except Exception: pass 
            return style_df
        
        st.markdown(df_z_lookup_table_mw.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                               {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_z_lookup_table_mw, axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows P(Z < z). Highlighted cell, row, and column are closest to the z-critical value derived from α={alpha_mw:.8f} and tail selection.")


    with col2: 
        st.subheader("P-value Calculation Explanation")
        p_val_calc_mw = float('nan')
        
        if tail_mw == "Two-tailed": p_val_calc_mw = 2 * stats.norm.sf(abs(z_calc_mw))
        elif tail_mw == "One-tailed (right)": p_val_calc_mw = stats.norm.sf(z_calc_mw)
        else:  p_val_calc_mw = stats.norm.cdf(z_calc_mw)
        p_val_calc_mw = min(p_val_calc_mw, 1.0) if not np.isnan(p_val_calc_mw) else float('nan')
        
        st.markdown(f"""
        The U statistic ({u_stat_mw:.1f}) is used to calculate an approximate z-statistic ({z_calc_mw:.3f}).
        The p-value is then derived from the standard normal distribution.
        * **Two-tailed**: `2 * P(Z ≥ |{z_calc_mw:.3f}|)`
        * **One-tailed (right)**: `P(Z ≥ {z_calc_mw:.3f})` 
        * **One-tailed (left)**: `P(Z ≤ {z_calc_mw:.3f})` 
        """)

        st.subheader("Summary")
        crit_val_z_display_mw = "N/A"
        if tail_mw == "Two-tailed": crit_val_z_display_mw = f"±{format_value_for_display(crit_z_upper_mw_plot)}"
        elif tail_mw == "One-tailed (right)": crit_val_z_display_mw = format_value_for_display(crit_z_upper_mw_plot)
        else: crit_val_z_display_mw = format_value_for_display(crit_z_lower_mw_plot)
        
        decision_crit_mw = False
        if crit_z_upper_mw_plot is not None: 
            if tail_mw == "Two-tailed": decision_crit_mw = abs(z_calc_mw) > crit_z_upper_mw_plot if not np.isnan(crit_z_upper_mw_plot) else False
            elif tail_mw == "One-tailed (right)": decision_crit_mw = z_calc_mw > crit_z_upper_mw_plot if not np.isnan(crit_z_upper_mw_plot) else False
        if crit_z_lower_mw_plot is not None and tail_mw == "One-tailed (left)":
             decision_crit_mw = z_calc_mw < crit_z_lower_mw_plot if not np.isnan(crit_z_lower_mw_plot) else False
        
        comparison_crit_str_mw = f"Approx. z_calc ({z_calc_mw:.3f}) vs z_crit ({crit_val_z_display_mw})"
        decision_p_alpha_mw = p_val_calc_mw < alpha_mw if not np.isnan(p_val_calc_mw) else False
        
        st.markdown(f"""
        1.  **Critical z-value ({tail_mw})**: {crit_val_z_display_mw}
            * *Associated p-value (α or α/2 per tail)*: {alpha_mw:.8f}
        2.  **Calculated U-statistic**: {u_stat_mw:.1f} (Approx. z-statistic: {z_calc_mw:.3f})
            * *Calculated p-value (Normal Approx.)*: {format_value_for_display(p_val_calc_mw, decimals=4)} ({apa_p_value(p_val_calc_mw)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_mw else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_mw}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_mw else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_mw)} is {'less than' if decision_p_alpha_mw else 'not less than'} α ({alpha_mw:.8f}).
        5.  **APA 7 Style Report (based on Normal Approximation)**:
            A Mann-Whitney U test (using normal approximation) indicated that the outcome for group 1 (n<sub>1</sub>={n1_mw}) was {'' if decision_p_alpha_mw else 'not '}statistically significantly different from group 2 (n<sub>2</sub>={n2_mw}), *U* = {u_stat_mw:.1f}, *z* = {z_calc_mw:.2f}, {apa_p_value(p_val_calc_mw)}. The null hypothesis was {'rejected' if decision_p_alpha_mw else 'not rejected'} at α = {alpha_mw:.2f}.
        """)

# --- Tab 6: Wilcoxon Signed-Rank T Test ---
def tab_wilcoxon_t():
    st.header("Wilcoxon Signed-Rank T Test (Normal Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_w = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_w_input")
        n_w = st.number_input("Sample Size (n, non-zero differences)", 1, 1000, 15, 1, key="n_w_input") 
        tail_w = st.radio("Tail Selection", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_w_radio")
        t_stat_w_input = st.number_input("Calculated T-statistic (sum of ranks)", value=float(n_w*(n_w+1)/4 / 2 if n_w >0 else 0), format="%.1f", min_value=0.0, max_value=float(n_w*(n_w+1)/2 if n_w > 0 else 0), key="t_stat_w_input")
        
        st.info("This tab uses the Normal Approximation for the Wilcoxon Signed-Rank T test. For small n (e.g., < 20), this is an approximation; exact methods/tables are preferred for definitive conclusions.")

        mu_t_w = n_w * (n_w + 1) / 4
        sigma_t_w_sq = n_w * (n_w + 1) * (2 * n_w + 1) / 24
        sigma_t_w = np.sqrt(sigma_t_w_sq) if sigma_t_w_sq > 0 else 0
        
        z_calc_w = 0.0
        if sigma_t_w > 0:
            if t_stat_w_input < mu_t_w: 
                z_calc_w = (t_stat_w_input + 0.5 - mu_t_w) / sigma_t_w
            elif t_stat_w_input > mu_t_w: 
                z_calc_w = (t_stat_w_input - 0.5 - mu_t_w) / sigma_t_w
        elif n_w > 0: 
            st.warning("Standard deviation (σ_T) for normal approximation is zero. Check sample size n. z_calc set to 0.")

        st.markdown(f"**Normal Approx. Parameters:** μ<sub>T</sub> = {mu_t_w:.2f}, σ<sub>T</sub> = {sigma_t_w:.2f}")
        st.markdown(f"**z-statistic (from T, for approx.):** {z_calc_w:.3f}")
        
        st.subheader("Distribution Plot (Normal Approximation)")
        fig_w, ax_w = plt.subplots(figsize=(8,5)); 
        plot_min_z_w = min(stats.norm.ppf(0.00000001), z_calc_w - 2, -4.0)
        plot_max_z_w = max(stats.norm.ppf(0.99999999), z_calc_w + 2, 4.0)
        if abs(z_calc_w) > 4 and abs(z_calc_w) > plot_max_z_w * 0.8:
             plot_min_z_w = min(plot_min_z_w, z_calc_w -1)
             plot_max_z_w = max(plot_max_z_w, z_calc_w +1)

        x_norm_w = np.linspace(plot_min_z_w, plot_max_z_w, 500)
        y_norm_w = stats.norm.pdf(x_norm_w)
        ax_w.plot(x_norm_w, y_norm_w, 'b-', lw=2, label='Standard Normal Distribution (Approx.)')
        
        crit_z_upper_w_plot, crit_z_lower_w_plot = None, None
        if tail_w == "Two-tailed": crit_z_upper_w_plot = stats.norm.ppf(1 - alpha_w / 2); crit_z_lower_w_plot = stats.norm.ppf(alpha_w / 2)
        elif tail_w == "One-tailed (right)": crit_z_upper_w_plot = stats.norm.ppf(1 - alpha_w)
        else: crit_z_lower_w_plot = stats.norm.ppf(alpha_w)

        if crit_z_upper_w_plot is not None and not np.isnan(crit_z_upper_w_plot):
            x_fill_upper = np.linspace(crit_z_upper_w_plot, plot_max_z_w, 100)
            ax_w.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, label=f'Approx. Crit. Region')
            ax_w.axvline(crit_z_upper_w_plot, color='red', linestyle='--', lw=1)
        if crit_z_lower_w_plot is not None and not np.isnan(crit_z_lower_w_plot):
            x_fill_lower = np.linspace(plot_min_z_w, crit_z_lower_w_plot, 100)
            ax_w.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.5, label=f'Approx. Crit. Region' if crit_z_upper_w_plot is None else None)
            ax_w.axvline(crit_z_lower_w_plot, color='red', linestyle='--', lw=1)

        ax_w.axvline(z_calc_w, color='green', linestyle='-', lw=2, label=f'Approx. z_calc = {z_calc_w:.3f}')
        ax_w.set_title('Normal Approx. for Wilcoxon T: Critical z Region(s)'); ax_w.legend(); ax_w.grid(True); st.pyplot(fig_w)

        st.subheader("Standard Normal Table: Cumulative P(Z < z)")
        st.markdown("This table shows the area to the left of a given z-score. The z-critical value (derived from your alpha and tail selection) is used for highlighting.")

        z_crit_for_table_w = crit_z_upper_w_plot if tail_w == "One-tailed (right)" else \
                             (crit_z_upper_w_plot if crit_z_upper_w_plot is not None and tail_w == "Two-tailed" else \
                             (crit_z_lower_w_plot if crit_z_lower_w_plot is not None else 0.0) )
        if z_crit_for_table_w is None or np.isnan(z_crit_for_table_w): z_crit_for_table_w = 0.0
        
        all_z_row_labels_w = [f"{val:.1f}" for val in np.round(np.arange(-3.4, 3.5, 0.1), 1)]
        z_col_labels_str_w = [f"{val:.2f}" for val in np.round(np.arange(0.00, 0.10, 0.01), 2)]
        
        z_target_for_table_row_numeric_w = round(z_crit_for_table_w, 1) 

        try:
            closest_row_idx_w = min(range(len(all_z_row_labels_w)), key=lambda i: abs(float(all_z_row_labels_w[i]) - z_target_for_table_row_numeric_w))
        except ValueError: 
            closest_row_idx_w = len(all_z_row_labels_w) // 2

        window_size_z_w = 5
        start_idx_z_w = max(0, closest_row_idx_w - window_size_z_w)
        end_idx_z_w = min(len(all_z_row_labels_w), closest_row_idx_w + window_size_z_w + 1)
        z_table_display_rows_str_w = all_z_row_labels_w[start_idx_z_w:end_idx_z_w]

        table_data_z_lookup_w = []
        for z_r_str_idx in z_table_display_rows_str_w:
            z_r_val = float(z_r_str_idx)
            row = { 'z': z_r_str_idx } 
            for z_c_str_idx in z_col_labels_str_w:
                z_c_val = float(z_c_str_idx)
                current_z_val = round(z_r_val + z_c_val, 2)
                prob = stats.norm.cdf(current_z_val)
                row[z_c_str_idx] = format_value_for_display(prob, decimals=4)
            table_data_z_lookup_w.append(row)
        
        df_z_lookup_table_w = pd.DataFrame(table_data_z_lookup_w).set_index('z')

        def style_z_lookup_table_w(df_to_style): # Reusing z-table styling logic
            data = df_to_style 
            style_df = pd.DataFrame('', index=data.index, columns=data.columns)
            z_crit_val_to_highlight = z_crit_for_table_w
            try:
                z_target_base_numeric = round(z_crit_val_to_highlight,1) 
                actual_row_labels_float = [float(label) for label in data.index]
                closest_row_float_val = min(actual_row_labels_float, key=lambda x_val: abs(x_val - z_target_base_numeric))
                highlight_row_label = f"{closest_row_float_val:.1f}"

                z_target_second_decimal_part = round(abs(z_crit_val_to_highlight - closest_row_float_val), 2) 
                actual_col_labels_float = [float(col_str) for col_str in data.columns]
                closest_col_float_val = min(actual_col_labels_float, key=lambda x_val: abs(x_val - z_target_second_decimal_part))
                highlight_col_label = f"{closest_col_float_val:.2f}"

                if highlight_row_label in style_df.index:
                    for col_name_iter in style_df.columns: 
                        style_df.loc[highlight_row_label, col_name_iter] = 'background-color: lightblue;'
                if highlight_col_label in style_df.columns:
                    for r_idx_iter in style_df.index: 
                        current_style = style_df.loc[r_idx_iter, highlight_col_label]
                        style_df.loc[r_idx_iter, highlight_col_label] = (current_style + ';' if current_style and not current_style.endswith(';') else current_style) + 'background-color: lightgreen;'
                if highlight_row_label in style_df.index and highlight_col_label in style_df.columns:
                    current_cell_style = style_df.loc[highlight_row_label, highlight_col_label]
                    style_df.loc[highlight_row_label, highlight_col_label] = (current_cell_style + ';' if current_cell_style and not current_cell_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            except Exception: pass 
            return style_df
        
        st.markdown(df_z_lookup_table_w.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                               {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_z_lookup_table_w, axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows P(Z < z). Highlighted cell, row, and column are closest to the z-critical value derived from α={alpha_w:.8f} and tail selection.")


    with col2: # Summary for Wilcoxon T
        st.subheader("P-value Calculation Explanation")
        p_val_calc_w = float('nan')
        
        if tail_w == "Two-tailed": p_val_calc_w = 2 * stats.norm.sf(abs(z_calc_w))
        elif tail_w == "One-tailed (right)": p_val_calc_w = stats.norm.sf(z_calc_w)
        else: p_val_calc_w = stats.norm.cdf(z_calc_w)
        p_val_calc_w = min(p_val_calc_w, 1.0) if not np.isnan(p_val_calc_w) else float('nan')

        st.markdown(f"""
        The T statistic ({t_stat_w_input:.1f}) is converted to an approximate z-statistic ({z_calc_w:.3f}) using μ<sub>T</sub>={mu_t_w:.2f}, σ<sub>T</sub>={sigma_t_w:.2f} (with continuity correction). P-value from normal distribution.
        * **Two-tailed**: `2 * P(Z ≥ |{z_calc_w:.3f}|)` 
        * **One-tailed (right)**: `P(Z ≥ {z_calc_w:.3f})` 
        * **One-tailed (left)**: `P(Z ≤ {z_calc_w:.3f})` 
        """)

        st.subheader("Summary")
        crit_val_display_w = "N/A"
        if tail_w == "Two-tailed": crit_val_display_w = f"±{format_value_for_display(crit_z_upper_w_plot)}"
        elif tail_w == "One-tailed (right)": crit_val_display_w = format_value_for_display(crit_z_upper_w_plot)
        else: crit_val_display_w = format_value_for_display(crit_z_lower_w_plot)

        decision_crit_w = False
        if crit_z_upper_w_plot is not None:
             if tail_w == "Two-tailed": decision_crit_w = abs(z_calc_w) > crit_z_upper_w_plot if not np.isnan(crit_z_upper_w_plot) else False
             elif tail_w == "One-tailed (right)": decision_crit_w = z_calc_w > crit_z_upper_w_plot if not np.isnan(crit_z_upper_w_plot) else False
        if crit_z_lower_w_plot is not None and tail_w == "One-tailed (left)":
             decision_crit_w = z_calc_w < crit_z_lower_w_plot if not np.isnan(crit_z_lower_w_plot) else False
        
        comparison_crit_str_w = f"Approx. z_calc ({z_calc_w:.3f}) vs z_crit ({crit_val_display_w})"
        decision_p_alpha_w = p_val_calc_w < alpha_w if not np.isnan(p_val_calc_w) else False
        
        st.markdown(f"""
        1.  **Critical z-value ({tail_w})**: {crit_val_display_w}
            * *Associated p-value (α or α/2 per tail)*: {alpha_w:.8f}
        2.  **Calculated T-statistic**: {t_stat_w_input:.1f} (Approx. z-statistic: {z_calc_w:.3f})
            * *Calculated p-value (Normal Approx.)*: {format_value_for_display(p_val_calc_w, decimals=4)} ({apa_p_value(p_val_calc_w)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_w else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_w}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_w else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_w)} is {'less than' if decision_p_alpha_w else 'not less than'} α ({alpha_w:.8f}).
        5.  **APA 7 Style Report (based on Normal Approximation)**:
            A Wilcoxon signed-rank test (using normal approximation) indicated that the median difference was {'' if decision_p_alpha_w else 'not '}statistically significant, *T* = {t_stat_w_input:.1f}, *z* = {z_calc_w:.2f}, {apa_p_value(p_val_calc_w)}. The null hypothesis was {'rejected' if decision_p_alpha_w else 'not rejected'} at α = {alpha_w:.2f} (n={n_w}).
        """)

# --- Tab 7: Binomial Test ---
def tab_binomial_test():
    st.header("Binomial Test Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_b = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_b_input")
        n_b = st.number_input("Number of Trials (n)", 1, 1000, 20, 1, key="n_b_input")
        p_null_b = st.number_input("Null Hypothesis Probability (p₀)", 0.00, 1.00, 0.5, 0.01, format="%.2f", key="p_null_b_input")
        k_success_b = st.number_input("Number of Successes (k)", 0, n_b, int(n_b * p_null_b), 1, key="k_success_b_input")
        
        tail_options_b = {
            f"Two-tailed (p ≠ {p_null_b})": "two-sided",
            f"One-tailed (right, p > {p_null_b})": "greater",
            f"One-tailed (left, p < {p_null_b})": "less"
        }
        tail_b_display = st.radio("Tail Selection (Alternative Hypothesis)", 
                                  list(tail_options_b.keys()), 
                                  key="tail_b_display_radio")
        tail_b_scipy = tail_options_b[tail_b_display]


        st.subheader("Binomial Distribution Plot")
        fig_b, ax_b = plt.subplots(figsize=(8,5))
        x_b_plot = np.arange(0, n_b + 1)
        y_b_pmf = stats.binom.pmf(x_b_plot, n_b, p_null_b)
        ax_b.bar(x_b_plot, y_b_pmf, label=f'Binomial PMF (n={n_b}, p₀={p_null_b})', alpha=0.7, color='skyblue')
        
        ax_b.scatter([k_success_b], [stats.binom.pmf(k_success_b, n_b, p_null_b)], color='green', s=100, zorder=5, label=f'Observed k = {k_success_b}')
        
        if tail_b_scipy == "greater": 
            crit_region_indices = x_b_plot[x_b_plot >= k_success_b]
            if len(crit_region_indices) > 0 :
                 ax_b.bar(crit_region_indices, y_b_pmf[x_b_plot >= k_success_b], color='salmon', alpha=0.6, label=f'P(X ≥ {k_success_b})')
        elif tail_b_scipy == "less": 
            crit_region_indices = x_b_plot[x_b_plot <= k_success_b]
            if len(crit_region_indices) > 0:
                ax_b.bar(crit_region_indices, y_b_pmf[x_b_plot <= k_success_b], color='salmon', alpha=0.6, label=f'P(X ≤ {k_success_b})')

        ax_b.set_title(f'Binomial Distribution (n={n_b}, p₀={p_null_b})')
        ax_b.set_xlabel('Number of Successes (k)')
        ax_b.set_ylabel('Probability Mass P(X=k)')
        ax_b.legend(); ax_b.grid(True); st.pyplot(fig_b)

        st.subheader("Probability Table Snippet")
        all_k_values_binomial = list(range(n_b + 1))
        table_k_window_binomial = get_dynamic_df_window(all_k_values_binomial, k_success_b, window_size=5)
        
        if len(table_k_window_binomial) > 0:
            table_data_b = {
                "k": table_k_window_binomial,
                "P(X=k)": [format_value_for_display(stats.binom.pmf(k_val, n_b, p_null_b), decimals=4) for k_val in table_k_window_binomial],
                "P(X≤k) (CDF)": [format_value_for_display(stats.binom.cdf(k_val, n_b, p_null_b), decimals=4) for k_val in table_k_window_binomial],
                "P(X≥k)": [format_value_for_display(stats.binom.sf(k_val -1, n_b, p_null_b), decimals=4) for k_val in table_k_window_binomial] 
            }
            df_table_b = pd.DataFrame(table_data_b)
            
            def highlight_k_row_b(row):
                if int(row["k"]) == k_success_b:
                    return ['background-color: yellow'] * len(row)
                return [''] * len(row)
            st.markdown(df_table_b.style.apply(highlight_k_row_b, axis=1).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows probabilities around k={k_success_b}. Highlighted row is your observed k. Column highlighting for alpha is not applicable to this table's structure.")
        else:
            st.info("Not enough range to display table snippet (e.g., n is very small).")
        st.markdown("""**Table Interpretation Note:** This table helps understand probabilities around the observed k. Critical regions for binomial tests are based on these cumulative probabilities compared to α.""")

    with col2: # Summary for Binomial Test
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value for a binomial test is the probability of observing k={k_success_b} successes, or results more extreme, given n={n_b} trials and null probability p₀={p_null_b}.
        * **Two-tailed (p ≠ {p_null_b})**: Sum of P(X=i) for all i where P(X=i) ≤ P(X={k_success_b}).
        * **One-tailed (right, p > {p_null_b})**: `P(X ≥ {k_success_b}) = stats.binom.sf({k_success_b}-1, n_b, p_null_b)`
        * **One-tailed (left, p < {p_null_b})**: `P(X ≤ {k_success_b}) = stats.binom.cdf({k_success_b}, n_b, p_null_b)`
        """)

        st.subheader("Summary")
        p_val_b_one_left = stats.binom.cdf(k_success_b, n_b, p_null_b)
        p_val_b_one_right = stats.binom.sf(k_success_b - 1, n_b, p_null_b) 

        if tail_b_scipy == "two-sided":
            p_observed_k = stats.binom.pmf(k_success_b, n_b, p_null_b)
            p_val_calc_b = 0
            for i in range(n_b + 1):
                if stats.binom.pmf(i, n_b, p_null_b) <= p_observed_k + 1e-9: # Tolerance
                    p_val_calc_b += stats.binom.pmf(i, n_b, p_null_b)
            p_val_calc_b = min(p_val_calc_b, 1.0) 
        elif tail_b_scipy == "greater":
            p_val_calc_b = p_val_b_one_right
        else: 
            p_val_calc_b = p_val_b_one_left
        
        crit_val_b_display = "Exact critical k values are complex for binomial; decision based on p-value."
        decision_p_alpha_b = p_val_calc_b < alpha_b
        decision_crit_b = decision_p_alpha_b 
        
        st.markdown(f"""
        1.  **Critical Region**: {crit_val_b_display}
            * *Significance level (α)*: {alpha_b:.8f}
        2.  **Observed Number of Successes (k)**: {k_success_b}
            * *Calculated p-value*: {format_value_for_display(p_val_calc_b, decimals=4)} ({apa_p_value(p_val_calc_b)})
        3.  **Decision (Critical Region Method - based on p-value)**: H₀ is **{'rejected' if decision_crit_b else 'not rejected'}**.
            * *Reason*: For discrete tests, the p-value method is generally preferred for decision making.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_b else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_b)} is {'less than' if decision_p_alpha_b else 'not less than'} α ({alpha_b:.8f}).
        5.  **APA 7 Style Report**:
            A binomial test was performed to assess whether the proportion of successes (k={k_success_b}, n={n_b}) was different from the null hypothesis proportion of p₀={p_null_b}. The result was {'' if decision_p_alpha_b else 'not '}statistically significant, {apa_p_value(p_val_calc_b)}. The null hypothesis was {'rejected' if decision_p_alpha_b else 'not rejected'} at α = {alpha_b:.2f}.
        """)


# --- Tab 8: Tukey HSD (Simplified Normal Approximation) ---
def tab_tukey_hsd():
    st.header("Tukey HSD (Simplified Normal Approximation)")
    col1, col2 = st.columns([2, 1.5])
    
    with col1:
        st.subheader("Inputs for Normal Approximation")
        alpha_tukey = st.number_input("Alpha (α) for z-comparison", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_tukey_approx_input")
        k_tukey_selected = st.number_input("Number of Groups (k) (for context)", min_value=2, value=3, step=1, key="k_tukey_context_selectbox") 
        df_error_tukey_selected = st.number_input("Degrees of Freedom for Error (df_error) (for context)", min_value=1, value=20, step=1, key="df_error_context_selectbox")
        
        test_stat_tukey_q_as_z = st.number_input("Your Calculated Statistic (q or other, treated as z-score)", value=1.0, format="%.3f", key="test_stat_tukey_q_as_z_input")
        
        tukey_tail_selection_approx = st.radio("Tail Selection for z-comparison", 
                                        ("Two-tailed", "One-tailed (right)"), 
                                        key="tukey_tail_z_approx_radio", index=1)

        st.warning("""
        **Important Note on Approximation:** This tab uses a **standard normal (z) distribution** to approximate 
        critical values and p-values. Your input 'Calculated Statistic' will be treated as a z-score for this comparison.
        This is a **significant simplification** and does **not** represent a true Tukey HSD test, which requires the 
        Studentized Range (q) distribution. For accurate Tukey HSD results, please use statistical software 
        that implements the Studentized Range distribution (e.g., `statsmodels` in Python, or R).
        """)

        st.subheader("Standard Normal (z) Distribution Plot")
        fig_tukey_approx, ax_tukey_approx = plt.subplots(figsize=(8,5))
        
        z_crit_upper_tukey_approx, z_crit_lower_tukey_approx = None, None
        
        if tukey_tail_selection_approx == "Two-tailed":
            z_crit_upper_tukey_approx = stats.norm.ppf(1 - alpha_tukey / 2)
            z_crit_lower_tukey_approx = stats.norm.ppf(alpha_tukey / 2)
        else: # One-tailed (right)
            z_crit_upper_tukey_approx = stats.norm.ppf(1 - alpha_tukey)

        plot_min_z_tukey = min(stats.norm.ppf(0.00000001), test_stat_tukey_q_as_z - 2, -4.0)
        plot_max_z_tukey = max(stats.norm.ppf(0.99999999), test_stat_tukey_q_as_z + 2, 4.0)
        if abs(test_stat_tukey_q_as_z) > 3.5 : 
             plot_min_z_tukey = min(plot_min_z_tukey, test_stat_tukey_q_as_z - 0.5)
             plot_max_z_tukey = max(plot_max_z_tukey, test_stat_tukey_q_as_z + 0.5)

        x_z_tukey_plot = np.linspace(plot_min_z_tukey, plot_max_z_tukey, 500)
        y_z_tukey_plot = stats.norm.pdf(x_z_tukey_plot)
        ax_tukey_approx.plot(x_z_tukey_plot, y_z_tukey_plot, 'b-', lw=2, label='Standard Normal Distribution (z)')

        if z_crit_upper_tukey_approx is not None and not np.isnan(z_crit_upper_tukey_approx):
            x_fill_upper = np.linspace(z_crit_upper_tukey_approx, plot_max_z_tukey, 100)
            ax_tukey_approx.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, 
                                  label=f'Crit. Region (α={alpha_tukey/2 if tukey_tail_selection_approx == "Two-tailed" else alpha_tukey:.8f})')
            ax_tukey_approx.axvline(z_crit_upper_tukey_approx, color='red', linestyle='--', lw=1)
        if z_crit_lower_tukey_approx is not None and not np.isnan(z_crit_lower_tukey_approx) and tukey_tail_selection_approx == "Two-tailed":
            x_fill_lower = np.linspace(plot_min_z_tukey, z_crit_lower_tukey_approx, 100)
            ax_tukey_approx.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.5)
            ax_tukey_approx.axvline(z_crit_lower_tukey_approx, color='red', linestyle='--', lw=1)
        
        ax_tukey_approx.axvline(test_stat_tukey_q_as_z, color='green', linestyle='-', lw=2, label=f'Input Stat (as z) = {test_stat_tukey_q_as_z:.3f}')
        ax_tukey_approx.set_title(f'Standard Normal Distribution for Approximation (α={alpha_tukey:.8f})')
        ax_tukey_approx.set_xlabel('Value (Treated as z-score)')
        ax_tukey_approx.set_ylabel('Probability Density')
        ax_tukey_approx.legend(); ax_tukey_approx.grid(True); st.pyplot(fig_tukey_approx)

        st.subheader("Standard Normal Table: Cumulative P(Z < z)")
        st.markdown("This table shows the area to the left of a given z-score. The z-critical value (derived from your alpha and tail selection) is used for highlighting.")

        z_crit_for_table_tukey = z_crit_upper_tukey_approx if tukey_tail_selection_approx == "One-tailed (right)" else \
                                (z_crit_upper_tukey_approx if z_crit_upper_tukey_approx is not None and tukey_tail_selection_approx == "Two-tailed" else \
                                (z_crit_lower_tukey_approx if z_crit_lower_tukey_approx is not None else 0.0) )
        if z_crit_for_table_tukey is None or np.isnan(z_crit_for_table_tukey): z_crit_for_table_tukey = 0.0


        all_z_row_labels_tukey = [f"{val:.1f}" for val in np.round(np.arange(-3.4, 3.5, 0.1), 1)]
        z_col_labels_str_tukey = [f"{val:.2f}" for val in np.round(np.arange(0.00, 0.10, 0.01), 2)]
        
        z_target_for_table_row_numeric_tukey = round(z_crit_for_table_tukey, 1) 

        try:
            closest_row_idx_tukey = min(range(len(all_z_row_labels_tukey)), key=lambda i: abs(float(all_z_row_labels_tukey[i]) - z_target_for_table_row_numeric_tukey))
        except ValueError: 
            closest_row_idx_tukey = len(all_z_row_labels_tukey) // 2

        window_size_z_tukey = 5
        start_idx_z_tukey = max(0, closest_row_idx_tukey - window_size_z_tukey)
        end_idx_z_tukey = min(len(all_z_row_labels_tukey), closest_row_idx_tukey + window_size_z_tukey + 1)
        z_table_display_rows_str_tukey = all_z_row_labels_tukey[start_idx_z_tukey:end_idx_z_tukey]

        table_data_z_lookup_tukey = []
        for z_r_str_idx in z_table_display_rows_str_tukey:
            z_r_val = float(z_r_str_idx)
            row = { 'z': z_r_str_idx } 
            for z_c_str_idx in z_col_labels_str_tukey:
                z_c_val = float(z_c_str_idx)
                current_z_val = round(z_r_val + z_c_val, 2)
                prob = stats.norm.cdf(current_z_val)
                row[z_c_str_idx] = format_value_for_display(prob, decimals=4)
            table_data_z_lookup_tukey.append(row)
        
        df_z_lookup_table_tukey = pd.DataFrame(table_data_z_lookup_tukey).set_index('z')

        def style_z_lookup_table_tukey(df_to_style):
            data = df_to_style 
            style_df = pd.DataFrame('', index=data.index, columns=data.columns)
            z_crit_val_to_highlight = z_crit_for_table_tukey
            
            try:
                z_target_base_numeric = round(z_crit_val_to_highlight,1) 
                actual_row_labels_float = [float(label) for label in data.index]
                closest_row_float_val = min(actual_row_labels_float, key=lambda x_val: abs(x_val - z_target_base_numeric))
                highlight_row_label = f"{closest_row_float_val:.1f}"

                z_target_second_decimal_part = round(abs(z_crit_val_to_highlight - closest_row_float_val), 2) 
                
                actual_col_labels_float = [float(col_str) for col_str in data.columns]
                closest_col_float_val = min(actual_col_labels_float, key=lambda x_val: abs(x_val - z_target_second_decimal_part))
                highlight_col_label = f"{closest_col_float_val:.2f}"

                if highlight_row_label in style_df.index:
                    for col_name_iter in style_df.columns: 
                        style_df.loc[highlight_row_label, col_name_iter] = 'background-color: lightblue;'
                
                if highlight_col_label in style_df.columns:
                    for r_idx_iter in style_df.index: 
                        current_style = style_df.loc[r_idx_iter, highlight_col_label]
                        style_df.loc[r_idx_iter, highlight_col_label] = (current_style + ';' if current_style and not current_style.endswith(';') else current_style) + 'background-color: lightgreen;'
                
                if highlight_row_label in style_df.index and highlight_col_label in style_df.columns:
                    current_cell_style = style_df.loc[highlight_row_label, highlight_col_label]
                    style_df.loc[highlight_row_label, highlight_col_label] = (current_cell_style + ';' if current_cell_style and not current_cell_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            
            except Exception: pass 
            return style_df
        
        st.markdown(df_z_lookup_table_tukey.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                               {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_z_lookup_table_tukey, axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows P(Z < z). Highlighted cell, row, and column are closest to the z-critical value derived from α={alpha_tukey:.8f} and tail selection.")


    with col2: 
        st.subheader("P-value Calculation Explanation (Normal Approximation)")
        p_val_calc_tukey_approx = float('nan')
        
        if tukey_tail_selection_approx == "Two-tailed":
            p_val_calc_tukey_approx = 2 * stats.norm.sf(abs(test_stat_tukey_q_as_z))
        else: # One-tailed (right)
            p_val_calc_tukey_approx = stats.norm.sf(test_stat_tukey_q_as_z)
        p_val_calc_tukey_approx = min(p_val_calc_tukey_approx, 1.0) if not np.isnan(p_val_calc_tukey_approx) else float('nan')

        st.markdown(f"""
        The input statistic ({test_stat_tukey_q_as_z:.3f}) is treated as a z-score.
        The p-value is then derived from the standard normal distribution:
        * **Two-tailed**: `2 * P(Z ≥ |{test_stat_tukey_q_as_z:.3f}|)`
        * **One-tailed (right)**: `P(Z ≥ {test_stat_tukey_q_as_z:.3f})`
        **Disclaimer**: This is a rough approximation and not a standard Tukey HSD p-value.
        """)

        st.subheader("Summary (Normal Approximation)")
        
        crit_val_tukey_display = "N/A"
        decision_crit_tukey_approx = False
        comparison_crit_str_tukey = "N/A"

        if tukey_tail_selection_approx == "Two-tailed":
            crit_val_tukey_display = f"±{format_value_for_display(z_crit_upper_tukey_approx)}" if z_crit_upper_tukey_approx is not None else "N/A"
            if z_crit_upper_tukey_approx is not None and not np.isnan(z_crit_upper_tukey_approx):
                decision_crit_tukey_approx = abs(test_stat_tukey_q_as_z) > z_crit_upper_tukey_approx
            comparison_crit_str_tukey = f"|Input Stat (as z) ({abs(test_stat_tukey_q_as_z):.3f})| {' > ' if decision_crit_tukey_approx else ' ≤ '} z_crit ({format_value_for_display(z_crit_upper_tukey_approx)})"
        else: # One-tailed (right)
            crit_val_tukey_display = format_value_for_display(z_crit_upper_tukey_approx) if z_crit_upper_tukey_approx is not None else "N/A"
            if z_crit_upper_tukey_approx is not None and not np.isnan(z_crit_upper_tukey_approx):
                decision_crit_tukey_approx = test_stat_tukey_q_as_z > z_crit_upper_tukey_approx
            comparison_crit_str_tukey = f"Input Stat (as z) ({test_stat_tukey_q_as_z:.3f}) {' > ' if decision_crit_tukey_approx else ' ≤ '} z_crit ({format_value_for_display(z_crit_upper_tukey_approx)})"

        decision_p_alpha_tukey_approx = p_val_calc_tukey_approx < alpha_tukey if not np.isnan(p_val_calc_tukey_approx) else False
            
        st.markdown(f"""
        1.  **Approximate Critical z-value ({tukey_tail_selection_approx})**: {crit_val_tukey_display}
            * *Significance level (α)*: {alpha_tukey:.8f}
        2.  **Input Statistic (treated as z-score)**: {test_stat_tukey_q_as_z:.3f}
            * *Approximate p-value (from z-dist)*: {format_value_for_display(p_val_calc_tukey_approx, decimals=4)} ({apa_p_value(p_val_calc_tukey_approx)})
        3.  **Decision (Approx. Critical Value Method)**: H₀ (no difference) is **{'rejected' if decision_crit_tukey_approx else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_tukey}.
        4.  **Decision (Approx. p-value Method)**: H₀ (no difference) is **{'rejected' if decision_p_alpha_tukey_approx else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_tukey_approx)} is {'less than' if decision_p_alpha_tukey_approx else 'not less than'} α ({alpha_tukey:.8f}).
        5.  **APA 7 Style Report (using Normal Approximation)**:
            Using a normal approximation for comparison, an input statistic of {test_stat_tukey_q_as_z:.2f} (for k={k_tukey_selected} groups, df<sub>error</sub>={df_error_tukey_selected}) yielded an approximate {apa_p_value(p_val_calc_tukey_approx)}. The null hypothesis of no difference for this pair was {'rejected' if decision_p_alpha_tukey_approx else 'not rejected'} at α = {alpha_tukey:.2f}. (Note: This is a z-distribution based approximation, not a standard Tukey HSD test).
        """)


# --- Tab 9: Kruskal-Wallis H Test ---
def tab_kruskal_wallis():
    st.header("Kruskal-Wallis H Test (Chi-square Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_kw = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_kw_input")
        k_groups_kw = st.number_input("Number of Groups (k)", 2, 50, 3, 1, key="k_groups_kw_input") 
        df_kw = k_groups_kw - 1
        st.markdown(f"Degrees of Freedom (df) = k - 1 = {df_kw}")
        test_stat_h_kw = st.number_input("Calculated H-statistic", value=float(df_kw if df_kw > 0 else 0.5), format="%.3f", min_value=0.0, key="test_stat_h_kw_input")
        st.caption("Note: Chi-square approximation is best if each group size ≥ 5.")

        st.subheader("Chi-square Distribution Plot (Approximation for H)")
        fig_kw, ax_kw = plt.subplots(figsize=(8,5))
        crit_val_chi2_kw_plot = None 
        
        if df_kw > 0:
            crit_val_chi2_kw_plot = stats.chi2.ppf(1 - alpha_kw, df_kw)
            plot_min_chi2_kw = 0.001
            plot_max_chi2_kw = max(stats.chi2.ppf(0.999, df_kw), test_stat_h_kw * 1.5, 10.0)
            if test_stat_h_kw > stats.chi2.ppf(0.999, df_kw) * 1.2:
                plot_max_chi2_kw = test_stat_h_kw * 1.2

            x_chi2_kw_plot = np.linspace(plot_min_chi2_kw, plot_max_chi2_kw, 500)
            y_chi2_kw_plot = stats.chi2.pdf(x_chi2_kw_plot, df_kw)
            ax_kw.plot(x_chi2_kw_plot, y_chi2_kw_plot, 'b-', lw=2, label=f'χ²-distribution (df={df_kw})')

            if isinstance(crit_val_chi2_kw_plot, (int, float)) and not np.isnan(crit_val_chi2_kw_plot):
                x_fill_upper_kw = np.linspace(crit_val_chi2_kw_plot, plot_max_chi2_kw, 100)
                ax_kw.fill_between(x_fill_upper_kw, stats.chi2.pdf(x_fill_upper_kw, df_kw), color='red', alpha=0.5, label=f'α = {alpha_kw:.8f}')
                ax_kw.axvline(crit_val_chi2_kw_plot, color='red', linestyle='--', lw=1, label=f'χ²_crit = {crit_val_chi2_kw_plot:.3f}')
            
            ax_kw.axvline(test_stat_h_kw, color='green', linestyle='-', lw=2, label=f'H_calc = {test_stat_h_kw:.3f}')
            ax_kw.set_title(f'χ²-Approximation for Kruskal-Wallis H (df={df_kw})')
        else:
            ax_kw.text(0.5, 0.5, "df must be > 0 (k > 1 for meaningful test)", ha='center', va='center')
            ax_kw.set_title('Plot Unavailable (df=0)')
            
        ax_kw.legend()
        ax_kw.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_kw)
        
        st.subheader("Critical χ²-Values (Upper Tail)")
        all_df_chi2_options_kw = list(range(1, 31)) + [35, 40, 45, 50, 60, 70, 80, 90, 100]
        table_df_window_chi2_kw = get_dynamic_df_window(all_df_chi2_options_kw, df_kw, window_size=5)
        table_alpha_cols_chi2 = [0.10, 0.05, 0.025, 0.01, 0.005]

        chi2_table_rows = []
        for df_iter_val in table_df_window_chi2_kw: # df_iter_val is already a number from window
            df_iter_calc = int(df_iter_val)
            row_data = {'df': str(df_iter_val)}
            for alpha_c in table_alpha_cols_chi2:
                cv = stats.chi2.ppf(1 - alpha_c, df_iter_calc)
                row_data[f"α = {alpha_c:.3f}"] = format_value_for_display(cv)
            chi2_table_rows.append(row_data)
        df_chi2_table_kw = pd.DataFrame(chi2_table_rows).set_index('df')

        def style_chi2_table_kw(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_str = str(df_kw) 

            if selected_df_str in df_to_style.index:
                style.loc[selected_df_str, :] = 'background-color: lightblue;'
            
            closest_alpha_col_val = min(table_alpha_cols_chi2, key=lambda x: abs(x - alpha_kw))
            highlight_col_name = f"α = {closest_alpha_col_val:.3f}"

            if highlight_col_name in df_to_style.columns:
                for r_idx in df_to_style.index:
                    current_r_style = style.loc[r_idx, highlight_col_name]
                    style.loc[r_idx, highlight_col_name] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                if selected_df_str in df_to_style.index:
                    current_c_style = style.loc[selected_df_str, highlight_col_name]
                    style.loc[selected_df_str, highlight_col_name] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style
        
        if df_kw > 0:
            st.markdown(df_chi2_table_kw.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                                {'selector': 'td', 'props': [('text-align', 'center')]}])
                                         .apply(style_chi2_table_kw, axis=None).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows upper-tail critical χ²-values. Highlighted for df={df_kw} and α closest to your test.")
        else:
            st.warning("df must be > 0 to generate table (k > 1).")


        st.markdown("""
        **Cumulative Table Note:** Kruskal-Wallis H is approx. χ² distributed (df = k-1). Test is right-tailed: large H suggests group differences.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value is P(χ² ≥ H_calc) assuming H₀ (all group medians are equal) is true.
        * `P(χ² ≥ {test_stat_h_kw:.3f}) = stats.chi2.sf({test_stat_h_kw:.3f}, df={df_kw})` (if df > 0)
        """)
        st.subheader("Summary")
        p_val_calc_kw_num = float('nan') 
        decision_crit_kw = False
        comparison_crit_str_kw = "Test not valid (df must be > 0)"
        decision_p_alpha_kw = False
        apa_H_stat = f"*H*({df_kw if df_kw > 0 else 'N/A'}) = {format_value_for_display(test_stat_h_kw, decimals=2)}"
        
        summary_crit_val_chi2_kw_display_str = "N/A (df=0)"
        if df_kw > 0:
            p_val_calc_kw_num = stats.chi2.sf(test_stat_h_kw, df_kw) 
            
            if isinstance(crit_val_chi2_kw_plot, (int, float)) and not np.isnan(crit_val_chi2_kw_plot):
                summary_crit_val_chi2_kw_display_str = format_value_for_display(crit_val_chi2_kw_plot)
                decision_crit_kw = test_stat_h_kw > crit_val_chi2_kw_plot
                comparison_crit_str_kw = f"H({format_value_for_display(test_stat_h_kw)}) > χ²_crit({format_value_for_display(crit_val_chi2_kw_plot)})" if decision_crit_kw else f"H({format_value_for_display(test_stat_h_kw)}) ≤ χ²_crit({format_value_for_display(crit_val_chi2_kw_plot)})"
            else:
                summary_crit_val_chi2_kw_display_str = "N/A (calc error)"
                comparison_crit_str_kw = "Comparison not possible (critical value is N/A or NaN)"
            
            if isinstance(p_val_calc_kw_num, (int, float)) and not np.isnan(p_val_calc_kw_num):
                decision_p_alpha_kw = p_val_calc_kw_num < alpha_kw
        
        apa_p_val_calc_kw_str = apa_p_value(p_val_calc_kw_num) 


        st.markdown(f"""
        1.  **Critical χ²-value (df={df_kw})**: {summary_crit_val_chi2_kw_display_str}
            * *Associated p-value (α)*: {alpha_kw:.8f}
        2.  **Calculated H-statistic**: {format_value_for_display(test_stat_h_kw)}
            * *Calculated p-value (from χ² approx.)*: {format_value_for_display(p_val_calc_kw_num, decimals=4)} ({apa_p_val_calc_kw_str})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_kw else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_kw}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_kw else 'not rejected'}**.
            * *Reason*: {apa_p_val_calc_kw_str} is {'less than' if decision_p_alpha_kw else 'not less than'} α ({alpha_kw:.8f}). 
        5.  **APA 7 Style Report**:
            A Kruskal-Wallis H test showed that there was {'' if decision_p_alpha_kw else 'not '}a statistically significant difference in medians between the k={k_groups_kw} groups, {apa_H_stat}, {apa_p_val_calc_kw_str}. The null hypothesis was {'rejected' if decision_p_alpha_kw else 'not rejected'} at α = {alpha_kw:.2f}.
        """)


# --- Tab 10: Friedman Test ---
def tab_friedman_test():
    st.header("Friedman Test (Chi-square Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_fr = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_fr_input")
        k_conditions_fr = st.number_input("Number of Conditions/Treatments (k)", 2, 50, 3, 1, key="k_conditions_fr_input") 
        n_blocks_fr = st.number_input("Number of Blocks/Subjects (n)", 2, 200, 10, 1, key="n_blocks_fr_input") 
        
        df_fr = k_conditions_fr - 1
        st.markdown(f"Degrees of Freedom (df) = k - 1 = {df_fr}")
        test_stat_q_fr = st.number_input("Calculated Friedman Q-statistic (or χ²_r)", value=float(df_fr if df_fr > 0 else 0.5), format="%.3f", min_value=0.0, key="test_stat_q_fr_input")

        if n_blocks_fr <= 10 or k_conditions_fr <= 3 : 
            st.warning("Small n or k. Friedman’s χ² approximation may be less reliable. Exact methods preferred if available.")

        st.subheader("Chi-square Distribution Plot (Approximation for Q)")
        fig_fr, ax_fr = plt.subplots(figsize=(8,5))
        crit_val_chi2_fr_plot = None 
        
        if df_fr > 0:
            crit_val_chi2_fr_plot = stats.chi2.ppf(1 - alpha_fr, df_fr)
            plot_min_chi2_fr = 0.001
            plot_max_chi2_fr = max(stats.chi2.ppf(0.999, df_fr), test_stat_q_fr * 1.5, 10.0)
            if test_stat_q_fr > stats.chi2.ppf(0.999, df_fr) * 1.2:
                plot_max_chi2_fr = test_stat_q_fr * 1.2

            x_chi2_fr_plot = np.linspace(plot_min_chi2_fr, plot_max_chi2_fr, 500)
            y_chi2_fr_plot = stats.chi2.pdf(x_chi2_fr_plot, df_fr)
            ax_fr.plot(x_chi2_fr_plot, y_chi2_fr_plot, 'b-', lw=2, label=f'χ²-distribution (df={df_fr})')
            
            if isinstance(crit_val_chi2_fr_plot, (int,float)) and not np.isnan(crit_val_chi2_fr_plot):
                x_fill_upper_fr = np.linspace(crit_val_chi2_fr_plot, plot_max_chi2_fr, 100)
                ax_fr.fill_between(x_fill_upper_fr, stats.chi2.pdf(x_fill_upper_fr, df_fr), color='red', alpha=0.5, label=f'α = {alpha_fr:.8f}')
                ax_fr.axvline(crit_val_chi2_fr_plot, color='red', linestyle='--', lw=1, label=f'χ²_crit = {crit_val_chi2_fr_plot:.3f}')
            
            ax_fr.axvline(test_stat_q_fr, color='green', linestyle='-', lw=2, label=f'Q_calc = {test_stat_q_fr:.3f}')
            ax_fr.set_title(f'χ²-Approximation for Friedman Q (df={df_fr})')
        else:
            ax_fr.text(0.5, 0.5, "df must be > 0 (k > 1 for meaningful test)", ha='center', va='center')
            ax_fr.set_title('Plot Unavailable (df=0)')
        ax_fr.legend()
        ax_fr.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_fr)

        st.subheader("Critical χ²-Values (Upper Tail)") # Same table as Kruskal-Wallis
        all_df_chi2_options_fr = list(range(1, 31)) + [35, 40, 45, 50, 60, 70, 80, 90, 100]
        table_df_window_chi2_fr = get_dynamic_df_window(all_df_chi2_options_fr, df_fr, window_size=5)
        table_alpha_cols_chi2_fr = [0.10, 0.05, 0.025, 0.01, 0.005]

        chi2_table_rows_fr = []
        for df_iter_val in table_df_window_chi2_fr: # df_iter_val is already a number
            df_iter_calc = int(df_iter_val)
            row_data = {'df': str(df_iter_val)}
            for alpha_c in table_alpha_cols_chi2_fr:
                cv = stats.chi2.ppf(1 - alpha_c, df_iter_calc)
                row_data[f"α = {alpha_c:.3f}"] = format_value_for_display(cv)
            chi2_table_rows_fr.append(row_data)
        df_chi2_table_fr = pd.DataFrame(chi2_table_rows_fr).set_index('df')

        def style_chi2_table_fr(df_to_style): # Similar styling to KW
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_str = str(df_fr) 
            
            if selected_df_str in df_to_style.index:
                style.loc[selected_df_str, :] = 'background-color: lightblue;'
            
            closest_alpha_col_val = min(table_alpha_cols_chi2_fr, key=lambda x: abs(x - alpha_fr))
            highlight_col_name = f"α = {closest_alpha_col_val:.3f}"

            if highlight_col_name in df_to_style.columns:
                for r_idx in df_to_style.index:
                    current_r_style = style.loc[r_idx, highlight_col_name]
                    style.loc[r_idx, highlight_col_name] = (current_r_style if current_r_style else '') + 'background-color: lightgreen;'
                if selected_df_str in df_to_style.index:
                    current_c_style = style.loc[selected_df_str, highlight_col_name]
                    style.loc[selected_df_str, highlight_col_name] = (current_c_style if current_c_style else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style
        
        if df_fr > 0:
            st.markdown(df_chi2_table_fr.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                                {'selector': 'td', 'props': [('text-align', 'center')]}])
                                         .apply(style_chi2_table_fr, axis=None).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows upper-tail critical χ²-values. Highlighted for df={df_fr} and α closest to your test.")
        else:
            st.warning("df must be > 0 to generate table (k > 1).")
        st.markdown("""
        **Cumulative Table Note:** Friedman Q statistic is approx. χ² distributed (df = k-1). Test is right-tailed.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        P-value is P(χ² ≥ Q_calc) assuming H₀ (all condition medians are equal across blocks) is true.
        * `P(χ² ≥ {test_stat_q_fr:.3f}) = stats.chi2.sf({test_stat_q_fr:.3f}, df={df_fr})` (if df > 0)
        """)

        st.subheader("Summary")
        p_val_calc_fr_num = float('nan') 
        decision_crit_fr = False
        comparison_crit_str_fr = "Test not valid (df must be > 0)"
        decision_p_alpha_fr = False
        apa_Q_stat = f"χ²<sub>r</sub>({df_fr if df_fr > 0 else 'N/A'}) = {format_value_for_display(test_stat_q_fr, decimals=2)}"
        
        summary_crit_val_chi2_fr_display_str = "N/A (df=0)"
        if df_fr > 0:
            p_val_calc_fr_num = stats.chi2.sf(test_stat_q_fr, df_fr) 
            if isinstance(crit_val_chi2_fr_plot, (int,float)) and not np.isnan(crit_val_chi2_fr_plot):
                summary_crit_val_chi2_fr_display_str = f"{crit_val_chi2_fr_plot:.3f}"
                decision_crit_fr = test_stat_q_fr > crit_val_chi2_fr_plot
                comparison_crit_str_fr = f"Q({test_stat_q_fr:.3f}) > χ²_crit({crit_val_chi2_fr_plot:.3f})" if decision_crit_fr else f"Q({test_stat_q_fr:.3f}) ≤ χ²_crit({crit_val_chi2_fr_plot:.3f})"
            else: 
                summary_crit_val_chi2_fr_display_str = "N/A (calc error)"
                comparison_crit_str_fr = "Comparison not possible (critical value is N/A or NaN)"

            if isinstance(p_val_calc_fr_num, (int, float)) and not np.isnan(p_val_calc_fr_num):
                decision_p_alpha_fr = p_val_calc_fr_num < alpha_fr
        
        apa_p_val_calc_fr_str = apa_p_value(p_val_calc_fr_num) 

        st.markdown(f"""
        1.  **Critical χ²-value (df={df_fr})**: {summary_crit_val_chi2_fr_display_str}
            * *Associated p-value (α)*: {alpha_fr:.8f}
        2.  **Calculated Q-statistic (χ²_r)**: {format_value_for_display(test_stat_q_fr)}
            * *Calculated p-value (from χ² approx.)*: {format_value_for_display(p_val_calc_fr_num, decimals=4)} ({apa_p_val_calc_fr_str})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_fr else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_fr}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_fr else 'not rejected'}**.
            * *Reason*: {apa_p_val_calc_fr_str} is {'less than' if decision_p_alpha_fr else 'not less than'} α ({alpha_fr:.8f}).
        5.  **APA 7 Style Report**:
            A Friedman test indicated that there was {'' if decision_p_alpha_fr else 'not '}a statistically significant difference in medians across the k={k_conditions_fr} conditions for n={n_blocks_fr} blocks, {apa_Q_stat}, {apa_p_val_calc_fr_str}. The null hypothesis was {'rejected' if decision_p_alpha_fr else 'not rejected'} at α = {alpha_fr:.2f}.
        """, unsafe_allow_html=True)

# --- Tab 11: Critical r Table (Pearson Correlation) ---
def tab_critical_r():
    st.header("Critical r Value Explorer (Pearson Correlation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_r_input = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_r_input")
        n_r = st.number_input("Sample Size (n, number of pairs)", min_value=3, value=20, step=1, key="n_r_input")
        test_stat_r = st.number_input("Your Calculated r-statistic", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, format="%.3f", key="test_stat_r_input")
        tail_r = st.radio("Tail Selection (H₁)", 
                          ("Two-tailed (ρ ≠ 0)", "One-tailed (positive, ρ > 0)", "One-tailed (negative, ρ < 0)"), 
                          key="tail_r_radio")

        df_r = n_r - 2
        if df_r <= 0:
            st.error("Sample size (n) must be greater than 2 to calculate degrees of freedom for correlation (df = n-2).")
            # Stop further execution in this tab if df is not valid
            st.stop() 
        
        st.markdown(f"**Degrees of Freedom (df)** = n - 2 = **{df_r}**")

        # Convert r to t for plotting and p-value
        t_observed_r = float('nan')
        if abs(test_stat_r) < 1.0: # Ensure 1-r^2 is not zero or negative
            try:
                if (1 - test_stat_r**2) <= 1e-9: # Handles r close to 1 or -1 edge case for sqrt
                    t_observed_r = float('inf') * np.sign(test_stat_r) if test_stat_r != 0 else 0.0
                else:
                    t_observed_r = test_stat_r * math.sqrt(df_r / (1 - test_stat_r**2))
            except ZeroDivisionError: 
                 t_observed_r = float('inf') * np.sign(test_stat_r) if test_stat_r != 0 else 0.0
            except ValueError: 
                 t_observed_r = float('nan')
        elif abs(test_stat_r) >= 1.0: # Handles r = 1 or r = -1
            t_observed_r = float('inf') * np.sign(test_stat_r)


        st.markdown(f"**Observed r converted to t-statistic (for plot & p-value):** {format_value_for_display(t_observed_r)}")


        st.subheader(f"t-Distribution Plot (df={df_r}) for r-to-t Transformed Value")
        fig_r, ax_r = plt.subplots(figsize=(8,5))
        
        dist_label_plot_r = f't-distribution (df={df_r})'
        crit_func_ppf_plot_r = lambda q_val: stats.t.ppf(q_val, df_r)
        crit_func_pdf_plot_r = lambda x_val: stats.t.pdf(x_val, df_r)
        
        plot_min_r_t = min(crit_func_ppf_plot_r(0.00000001), (t_observed_r - 3) if not (np.isnan(t_observed_r) or np.isinf(t_observed_r)) else -3, -4.0)
        plot_max_r_t = max(crit_func_ppf_plot_r(0.99999999), (t_observed_r + 3) if not (np.isnan(t_observed_r) or np.isinf(t_observed_r)) else 3, 4.0)

        if not (np.isnan(t_observed_r) or np.isinf(t_observed_r)) and abs(t_observed_r) > 4 and abs(t_observed_r) > plot_max_r_t * 0.8 : 
            plot_min_r_t = min(plot_min_r_t, t_observed_r -1)
            plot_max_r_t = max(plot_max_r_t, t_observed_r +1)
        elif np.isinf(t_observed_r): 
            plot_min_r_t = -7 if t_observed_r < 0 else crit_func_ppf_plot_r(0.0001) 
            plot_max_r_t = 7 if t_observed_r > 0 else crit_func_ppf_plot_r(0.9999)


        x_r_t_plot = np.linspace(plot_min_r_t, plot_max_r_t, 500)
        y_r_t_plot = crit_func_pdf_plot_r(x_r_t_plot)
        ax_r.plot(x_r_t_plot, y_r_t_plot, 'b-', lw=2, label=dist_label_plot_r)

        crit_t_upper_r_plot, crit_t_lower_r_plot = None, None
        alpha_for_plot_r = alpha_r_input

        if tail_r == "Two-tailed (ρ ≠ 0)":
            crit_t_upper_r_plot = crit_func_ppf_plot_r(1 - alpha_for_plot_r / 2)
            crit_t_lower_r_plot = crit_func_ppf_plot_r(alpha_for_plot_r / 2)
            if crit_t_upper_r_plot is not None and not np.isnan(crit_t_upper_r_plot):
                 x_fill_upper = np.linspace(crit_t_upper_r_plot, plot_max_r_t, 100)
                 ax_r.fill_between(x_fill_upper, crit_func_pdf_plot_r(x_fill_upper), color='red', alpha=0.5, label=f'α/2 = {alpha_for_plot_r/2:.8f}')
                 ax_r.axvline(crit_t_upper_r_plot, color='red', linestyle='--', lw=1)
            if crit_t_lower_r_plot is not None and not np.isnan(crit_t_lower_r_plot):
                 x_fill_lower = np.linspace(plot_min_r_t, crit_t_lower_r_plot, 100)
                 ax_r.fill_between(x_fill_lower, crit_func_pdf_plot_r(x_fill_lower), color='red', alpha=0.5)
                 ax_r.axvline(crit_t_lower_r_plot, color='red', linestyle='--', lw=1)
        elif tail_r == "One-tailed (positive, ρ > 0)":
            crit_t_upper_r_plot = crit_func_ppf_plot_r(1 - alpha_for_plot_r)
            if crit_t_upper_r_plot is not None and not np.isnan(crit_t_upper_r_plot):
                x_fill_upper = np.linspace(crit_t_upper_r_plot, plot_max_r_t, 100)
                ax_r.fill_between(x_fill_upper, crit_func_pdf_plot_r(x_fill_upper), color='red', alpha=0.5, label=f'α = {alpha_for_plot_r:.8f}')
                ax_r.axvline(crit_t_upper_r_plot, color='red', linestyle='--', lw=1)
        else: # One-tailed (negative, ρ < 0)
            crit_t_lower_r_plot = crit_func_ppf_plot_r(alpha_for_plot_r)
            if crit_t_lower_r_plot is not None and not np.isnan(crit_t_lower_r_plot):
                x_fill_lower = np.linspace(plot_min_r_t, crit_t_lower_r_plot, 100)
                ax_r.fill_between(x_fill_lower, crit_func_pdf_plot_r(x_fill_lower), color='red', alpha=0.5, label=f'α = {alpha_for_plot_r:.8f}')
                ax_r.axvline(crit_t_lower_r_plot, color='red', linestyle='--', lw=1)
        
        if not np.isnan(t_observed_r) and not np.isinf(t_observed_r): # Only plot if finite
            ax_r.axvline(t_observed_r, color='green', linestyle='-', lw=2, label=f'Observed t (from r) = {t_observed_r:.3f}')
        
        ax_r.set_title(f't-Distribution for r (df={df_r})')
        ax_r.set_xlabel('t-value')
        ax_r.set_ylabel('Probability Density')
        ax_r.legend(); ax_r.grid(True); st.pyplot(fig_r)

        st.subheader("Critical r-Values Table (Two-tailed)")
        all_df_r_options = list(range(1, 101)) + [120, 150, 200, 300, 400, 500, 1000] 
        table_df_r_window = get_dynamic_df_window(all_df_r_options, df_r, window_size=5)
        table_alpha_r_cols = [0.10, 0.05, 0.02, 0.01, 0.001] 

        r_table_rows = []
        for df_iter in table_df_r_window:
            df_iter_calc = int(df_iter)
            if df_iter_calc <= 0: continue 
            row_data = {'df (n-2)': str(df_iter_calc)}
            for alpha_col in table_alpha_r_cols:
                t_crit_cell = stats.t.ppf(1 - alpha_col / 2, df_iter_calc) 
                r_crit_cell = float('nan')
                if not np.isnan(t_crit_cell) and (t_crit_cell**2 + df_iter_calc) != 0:
                    term_under_sqrt = t_crit_cell**2 / (t_crit_cell**2 + df_iter_calc)
                    if term_under_sqrt >= 0:
                         r_crit_cell = math.sqrt(term_under_sqrt)

                row_data[f"α (2-tail) = {alpha_col:.3f}"] = format_value_for_display(r_crit_cell, decimals=4)
            r_table_rows.append(row_data)
        
        df_r_table = pd.DataFrame(r_table_rows).set_index('df (n-2)')

        def style_r_table(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_r_str = str(df_r)

            if selected_df_r_str in df_to_style.index: 
                style.loc[selected_df_r_str, :] = 'background-color: lightblue;'
            
            effective_table_alpha = alpha_r_input if tail_r == "Two-tailed (ρ ≠ 0)" else alpha_r_input * 2
            
            closest_alpha_col_val_r = min(table_alpha_r_cols, key=lambda x: abs(x - effective_table_alpha))
            highlight_col_name_r = f"α (2-tail) = {closest_alpha_col_val_r:.3f}"

            if highlight_col_name_r in df_to_style.columns:
                for r_idx in df_to_style.index:
                     current_r_style = style.loc[r_idx, highlight_col_name_r]
                     style.loc[r_idx, highlight_col_name_r] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                if selected_df_r_str in df_to_style.index: 
                    current_c_style = style.loc[selected_df_r_str, highlight_col_name_r]
                    style.loc[selected_df_r_str, highlight_col_name_r] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style
        
        if not df_r_table.empty:
            st.markdown(df_r_table.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                          {'selector': 'td', 'props': [('text-align', 'center')]}])
                                         .apply(style_r_table, axis=None).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows two-tailed critical r-values. Highlighted for df={df_r} and α (2-tail) closest to your test's equivalent two-tailed α.")
        else:
            st.warning("Critical r-value table could not be generated for the current df.")
        st.markdown("""
        **Table Interpretation Note:**
        * This table displays critical r-values for **two-tailed tests**.
        * For **one-tailed tests**, you can use this table but adjust the alpha level. For example, a one-tailed test at α=0.05 uses the critical r from the α=0.10 (two-tailed) column.
        """)

    with col2:
        st.subheader("Significance Test for Pearson's r")
        
        p_val_r = float('nan')
        if not np.isnan(t_observed_r) and df_r > 0:
            if tail_r == "Two-tailed (ρ ≠ 0)":
                p_val_r = 2 * stats.t.sf(abs(t_observed_r), df_r)
            elif tail_r == "One-tailed (positive, ρ > 0)":
                p_val_r = stats.t.sf(t_observed_r, df_r)
            else: # One-tailed (negative, ρ < 0)
                p_val_r = stats.t.cdf(t_observed_r, df_r)
            p_val_r = min(p_val_r, 1.0) if not np.isnan(p_val_r) else float('nan')

        # Calculate precise critical r for summary
        t_crit_summary = float('nan')
        if df_r > 0:
            if tail_r == "Two-tailed (ρ ≠ 0)":
                t_crit_summary = stats.t.ppf(1 - alpha_r_input / 2, df_r)
            elif tail_r == "One-tailed (positive, ρ > 0)" or tail_r == "One-tailed (negative, ρ < 0)":
                 t_crit_summary = stats.t.ppf(1 - alpha_r_input, df_r) 
        
        r_crit_summary = float('nan')
        if not np.isnan(t_crit_summary) and df_r > 0 and (t_crit_summary**2 + df_r) != 0:
            term_under_sqrt_summary = t_crit_summary**2 / (t_crit_summary**2 + df_r)
            if term_under_sqrt_summary >=0:
                 r_crit_summary = math.sqrt(term_under_sqrt_summary)
        
        crit_r_display_summary = format_value_for_display(r_crit_summary, decimals=4)
        if tail_r == "Two-tailed (ρ ≠ 0)":
            crit_r_display_summary = f"±{crit_r_display_summary}"


        decision_crit_r = False
        comparison_crit_str_r = f"|{test_stat_r:.3f}| vs |{format_value_for_display(r_crit_summary, decimals=4)}|"
        if not np.isnan(r_crit_summary):
            if tail_r == "Two-tailed (ρ ≠ 0)":
                decision_crit_r = abs(test_stat_r) > r_crit_summary
                comparison_crit_str_r = f"|{test_stat_r:.3f}| ({abs(test_stat_r):.3f}) {' > ' if decision_crit_r else ' ≤ '} |r_crit| ({format_value_for_display(r_crit_summary, decimals=4)})"
            elif tail_r == "One-tailed (positive, ρ > 0)":
                decision_crit_r = test_stat_r > r_crit_summary
                comparison_crit_str_r = f"{test_stat_r:.3f} {' > ' if decision_crit_r else ' ≤ '} r_crit ({format_value_for_display(r_crit_summary, decimals=4)})"
            else: # One-tailed (negative, ρ < 0)
                decision_crit_r = test_stat_r < -r_crit_summary 
                comparison_crit_str_r = f"{test_stat_r:.3f} {' < ' if decision_crit_r else ' ≥ '} -r_crit (-{format_value_for_display(r_crit_summary, decimals=4)})"
        
        decision_p_alpha_r = p_val_r < alpha_r_input if not np.isnan(p_val_r) else False
        
        st.markdown(f"""
        1.  **Critical r-value ({tail_r})**: {crit_r_display_summary}
            * *Significance level (α)*: {alpha_r_input:.8f}
            * *Degrees of freedom (df = n-2)*: {df_r}
        2.  **Calculated r-statistic**: {test_stat_r:.3f} (corresponds to t = {format_value_for_display(t_observed_r)})
            * *Calculated p-value*: {format_value_for_display(p_val_r, decimals=4)} ({apa_p_value(p_val_r)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_r else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_r}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_r else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_r)} is {'less than' if decision_p_alpha_r else 'not less than'} α ({alpha_r_input:.8f}).
        5.  **APA 7 Style Report**:
            A Pearson correlation coefficient was computed to assess the linear relationship between the two variables. The correlation was found to be *r*({df_r}) = {test_stat_r:.2f}, {apa_p_value(p_val_r)}. The null hypothesis was {'rejected' if decision_p_alpha_r else 'not rejected'} at the α = {alpha_r_input:.2f} level.
        """)

# --- Tab 12: Spearman’s Rank Correlation Critical Values Table ---
def tab_spearmans_rho():
    st.header("Spearman's Rank Correlation (ρ) Critical Values")
    st.markdown("""
    Tests the significance of Spearman's rank correlation coefficient (rho, *r<sub>s</sub>*). 
    The test statistic is often approximated using the t-distribution for sample sizes (n) greater than ~10-20.
    This tab uses the t-approximation. For very small sample sizes, exact critical value tables for Spearman's ρ should be consulted.
    """)
    
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_sr = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_sr_input")
        n_sr = st.number_input("Sample Size (n, number of pairs)", min_value=4, value=20, step=1, key="n_sr_input") # df = n-2, so n must be > 2. For table, often starts at n=4 or 5.
        test_stat_rho = st.number_input("Your Calculated Spearman's ρ (rho)", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, format="%.3f", key="test_stat_rho_input")
        tail_sr = st.radio("Tail Selection (H₁)", 
                          ("Two-tailed (ρ ≠ 0)", "One-tailed (positive, ρ > 0)", "One-tailed (negative, ρ < 0)"), 
                          key="tail_sr_radio")

        df_sr = n_sr - 2
        if df_sr <= 0: # Should be caught by min_value of n_sr, but as a safeguard
            st.error("Sample size (n) must be greater than 2 for the t-approximation (df = n-2).")
            st.stop()
        
        st.markdown(f"**Degrees of Freedom (df)** = n - 2 = **{df_sr}**")

        # Convert Spearman's rho to t-statistic
        t_observed_sr = float('nan')
        if abs(test_stat_rho) < 1.0: 
            try:
                if (1 - test_stat_rho**2) <= 1e-9: 
                     t_observed_sr = float('inf') * np.sign(test_stat_rho) if test_stat_rho !=0 else 0.0
                else:
                    t_observed_sr = test_stat_rho * math.sqrt(df_sr / (1 - test_stat_rho**2))
            except (ZeroDivisionError, ValueError):
                 t_observed_sr = float('inf') * np.sign(test_stat_rho) if test_stat_rho !=0 else float('nan')
        elif abs(test_stat_rho) >= 1.0: 
            t_observed_sr = float('inf') * np.sign(test_stat_rho)

        st.markdown(f"**Observed ρ converted to t-statistic (approx.):** {format_value_for_display(t_observed_sr)}")

        st.subheader(f"t-Distribution Plot (df={df_sr}) for ρ-to-t Transformed Value")
        fig_sr, ax_sr = plt.subplots(figsize=(8,5))
        
        dist_label_plot_sr = f't-distribution (df={df_sr})'
        
        plot_min_sr_t = min(stats.t.ppf(0.00000001, df_sr) if df_sr > 0 else -5, (t_observed_sr - 3) if not (np.isnan(t_observed_sr) or np.isinf(t_observed_sr)) else -3, -4.0)
        plot_max_sr_t = max(stats.t.ppf(0.99999999, df_sr) if df_sr > 0 else 5, (t_observed_sr + 3) if not (np.isnan(t_observed_sr) or np.isinf(t_observed_sr)) else 3, 4.0)
        if not (np.isnan(t_observed_sr) or np.isinf(t_observed_sr)) and abs(t_observed_sr) > 4 and abs(t_observed_sr) > plot_max_sr_t * 0.8 : 
            plot_min_sr_t = min(plot_min_sr_t, t_observed_sr -1)
            plot_max_sr_t = max(plot_max_sr_t, t_observed_sr +1)
        elif np.isinf(t_observed_sr) and df_sr > 0: 
            plot_min_sr_t = -7 if t_observed_sr < 0 else stats.t.ppf(0.0001, df_sr) 
            plot_max_sr_t = 7 if t_observed_sr > 0 else stats.t.ppf(0.9999, df_sr)

        x_sr_t_plot = np.linspace(plot_min_sr_t, plot_max_sr_t, 500)
        y_sr_t_plot = stats.t.pdf(x_sr_t_plot, df_sr) if df_sr > 0 else np.zeros_like(x_sr_t_plot)
        ax_sr.plot(x_sr_t_plot, y_sr_t_plot, 'b-', lw=2, label=dist_label_plot_sr)

        crit_t_upper_sr_plot, crit_t_lower_sr_plot = None, None
        if df_sr > 0:
            if tail_sr == "Two-tailed (ρ ≠ 0)":
                crit_t_upper_sr_plot = stats.t.ppf(1 - alpha_sr / 2, df_sr)
                crit_t_lower_sr_plot = stats.t.ppf(alpha_sr / 2, df_sr)
            elif tail_sr == "One-tailed (positive, ρ > 0)":
                crit_t_upper_sr_plot = stats.t.ppf(1 - alpha_sr, df_sr)
            else: # One-tailed (negative, ρ < 0)
                crit_t_lower_sr_plot = stats.t.ppf(alpha_sr, df_sr)

            if crit_t_upper_sr_plot is not None and not np.isnan(crit_t_upper_sr_plot):
                 x_fill_upper = np.linspace(crit_t_upper_sr_plot, plot_max_sr_t, 100)
                 ax_sr.fill_between(x_fill_upper, stats.t.pdf(x_fill_upper, df_sr), color='red', alpha=0.5, label=f'α/2 or α = {alpha_sr/(2 if tail_sr == "Two-tailed (ρ ≠ 0)" else 1):.8f}')
                 ax_sr.axvline(crit_t_upper_sr_plot, color='red', linestyle='--', lw=1)
            if crit_t_lower_sr_plot is not None and not np.isnan(crit_t_lower_sr_plot):
                 x_fill_lower = np.linspace(plot_min_sr_t, crit_t_lower_sr_plot, 100)
                 ax_sr.fill_between(x_fill_lower, stats.t.pdf(x_fill_lower, df_sr), color='red', alpha=0.5)
                 ax_sr.axvline(crit_t_lower_sr_plot, color='red', linestyle='--', lw=1)
        
        if not np.isnan(t_observed_sr) and not np.isinf(t_observed_sr):
            ax_sr.axvline(t_observed_sr, color='green', linestyle='-', lw=2, label=f'Observed t (from ρ) = {t_observed_sr:.3f}')
        
        ax_sr.set_title(f't-Distribution for Spearman\'s ρ (df={df_sr})')
        ax_sr.set_xlabel('t-value')
        ax_sr.set_ylabel('Probability Density')
        ax_sr.legend(); ax_sr.grid(True); st.pyplot(fig_sr)

        st.subheader("Critical Spearman's ρ Values Table (Two-tailed, t-approximation)")
        all_df_sr_options = list(range(1, 101)) + [120, 150, 200, 300, 400, 500, 1000] 
        table_df_sr_window = get_dynamic_df_window(all_df_sr_options, df_sr, window_size=5)
        table_alpha_sr_cols = [0.10, 0.05, 0.02, 0.01, 0.001] 

        sr_table_rows = []
        for df_iter in table_df_sr_window:
            df_iter_calc = int(df_iter)
            if df_iter_calc <= 0: continue 
            row_data = {'df (n-2)': str(df_iter_calc)}
            for alpha_col in table_alpha_sr_cols:
                t_crit_cell = stats.t.ppf(1 - alpha_col / 2, df_iter_calc) 
                rho_crit_cell = float('nan')
                if not np.isnan(t_crit_cell) and (t_crit_cell**2 + df_iter_calc) != 0:
                    term_under_sqrt = t_crit_cell**2 / (t_crit_cell**2 + df_iter_calc)
                    if term_under_sqrt >=0: 
                         rho_crit_cell = math.sqrt(term_under_sqrt)
                row_data[f"α (2-tail) = {alpha_col:.3f}"] = format_value_for_display(rho_crit_cell, decimals=4)
            sr_table_rows.append(row_data)
        
        df_sr_table = pd.DataFrame(sr_table_rows).set_index('df (n-2)')

        def style_sr_table(df_to_style): 
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_sr_str = str(df_sr)

            if selected_df_sr_str in df_to_style.index: 
                style.loc[selected_df_sr_str, :] = 'background-color: lightblue;'
            
            effective_table_alpha_sr = alpha_sr if tail_sr == "Two-tailed (ρ ≠ 0)" else alpha_sr * 2
            
            closest_alpha_col_val_sr = min(table_alpha_sr_cols, key=lambda x: abs(x - effective_table_alpha_sr))
            highlight_col_name_sr = f"α (2-tail) = {closest_alpha_col_val_sr:.3f}"

            if highlight_col_name_sr in df_to_style.columns:
                for r_idx in df_to_style.index:
                     current_r_style = style.loc[r_idx, highlight_col_name_sr]
                     style.loc[r_idx, highlight_col_name_sr] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                if selected_df_sr_str in df_to_style.index: 
                    current_c_style = style.loc[selected_df_sr_str, highlight_col_name_sr]
                    style.loc[selected_df_sr_str, highlight_col_name_sr] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style
        
        if not df_sr_table.empty:
            st.markdown(df_sr_table.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                          {'selector': 'td', 'props': [('text-align', 'center')]}])
                                         .apply(style_sr_table, axis=None).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows two-tailed critical Spearman's ρ values (using t-approximation). Highlighted for df={df_sr} and α (2-tail) closest to your test's equivalent two-tailed α.")
        else:
            st.warning("Critical Spearman's ρ table could not be generated for the current df.")
        st.markdown("""
        **Table Interpretation Note:**
        * This table displays critical ρ values for **two-tailed tests** using the t-approximation.
        * For **one-tailed tests**, you can use this table but adjust the alpha level (e.g., a one-tailed test at α=0.05 uses the critical ρ from the α=0.10 two-tailed column).
        """)

    with col2:
        st.subheader("Significance Test for Spearman's ρ")
        
        p_val_sr = float('nan')
        if not np.isnan(t_observed_sr) and df_sr > 0:
            if tail_sr == "Two-tailed (ρ ≠ 0)":
                p_val_sr = 2 * stats.t.sf(abs(t_observed_sr), df_sr)
            elif tail_sr == "One-tailed (positive, ρ > 0)":
                p_val_sr = stats.t.sf(t_observed_sr, df_sr)
            else: # One-tailed (negative, ρ < 0)
                p_val_sr = stats.t.cdf(t_observed_sr, df_sr)
            p_val_sr = min(p_val_sr, 1.0) if not np.isnan(p_val_sr) else float('nan')

        t_crit_summary_sr = float('nan')
        if df_sr > 0:
            if tail_sr == "Two-tailed (ρ ≠ 0)":
                t_crit_summary_sr = stats.t.ppf(1 - alpha_sr / 2, df_sr)
            elif tail_sr == "One-tailed (positive, ρ > 0)" or tail_sr == "One-tailed (negative, ρ < 0)":
                 t_crit_summary_sr = stats.t.ppf(1 - alpha_sr, df_sr) 
        
        rho_crit_summary = float('nan')
        if not np.isnan(t_crit_summary_sr) and df_sr > 0 and (t_crit_summary_sr**2 + df_sr) != 0:
            term_under_sqrt_summary_sr = t_crit_summary_sr**2 / (t_crit_summary_sr**2 + df_sr)
            if term_under_sqrt_summary_sr >=0:
                 rho_crit_summary = math.sqrt(term_under_sqrt_summary_sr)
        
        crit_rho_display_summary = format_value_for_display(rho_crit_summary, decimals=4)
        if tail_sr == "Two-tailed (ρ ≠ 0)":
            crit_rho_display_summary = f"±{crit_rho_display_summary}"

        decision_crit_sr = False
        comparison_crit_str_sr = f"|{test_stat_rho:.3f}| vs |{format_value_for_display(rho_crit_summary, decimals=4)}|"
        if not np.isnan(rho_crit_summary):
            if tail_sr == "Two-tailed (ρ ≠ 0)":
                decision_crit_sr = abs(test_stat_rho) > rho_crit_summary
                comparison_crit_str_sr = f"|{test_stat_rho:.3f}| ({abs(test_stat_rho):.3f}) {' > ' if decision_crit_sr else ' ≤ '} |ρ_crit| ({format_value_for_display(rho_crit_summary, decimals=4)})"
            elif tail_sr == "One-tailed (positive, ρ > 0)":
                decision_crit_sr = test_stat_rho > rho_crit_summary
                comparison_crit_str_sr = f"{test_stat_rho:.3f} {' > ' if decision_crit_sr else ' ≤ '} ρ_crit ({format_value_for_display(rho_crit_summary, decimals=4)})"
            else: # One-tailed (negative, ρ < 0)
                decision_crit_sr = test_stat_rho < -rho_crit_summary 
                comparison_crit_str_sr = f"{test_stat_rho:.3f} {' < ' if decision_crit_sr else ' ≥ '} -ρ_crit (-{format_value_for_display(rho_crit_summary, decimals=4)})"
        
        decision_p_alpha_sr = p_val_sr < alpha_sr if not np.isnan(p_val_sr) else False
        
        st.markdown(f"""
        1.  **Critical ρ-value ({tail_sr})**: {crit_rho_display_summary}
            * *Significance level (α)*: {alpha_sr:.8f}
            * *Degrees of freedom (df = n-2)*: {df_sr}
        2.  **Calculated ρ-statistic**: {test_stat_rho:.3f} (corresponds to t ≈ {format_value_for_display(t_observed_sr)})
            * *Calculated p-value (t-approx.)*: {format_value_for_display(p_val_sr, decimals=4)} ({apa_p_value(p_val_sr)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_sr else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_sr}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_sr else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_sr)} is {'less than' if decision_p_alpha_sr else 'not less than'} α ({alpha_sr:.8f}).
        5.  **APA 7 Style Report**:
            A Spearman's rank-order correlation was run to determine the relationship between the two variables. There was a {'' if decision_p_alpha_sr else 'non-'}statistically significant correlation, *rs*({df_sr}) = {test_stat_rho:.2f}, {apa_p_value(p_val_sr)}. The null hypothesis was {'rejected' if decision_p_alpha_sr else 'not rejected'} at α = {alpha_sr:.2f}.
        """)

# --- Tab 13: Kendall's Tau Critical Values Table ---
def tab_kendalls_tau():
    st.header("Kendall's Tau (τ) Significance Test (Normal Approximation)")
    st.markdown("Tests the significance of Kendall's rank correlation coefficient (tau). For n > ~10, a normal approximation is often used.")
    
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_tau = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_tau_input")
        n_tau = st.number_input("Sample Size (n, number of pairs)", min_value=4, value=20, step=1, key="n_tau_input") # Normal approx better for n > 10
        test_stat_tau = st.number_input("Your Calculated Kendall's τ (tau)", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, format="%.3f", key="test_stat_tau_input")
        tail_tau = st.radio("Tail Selection (H₁)", 
                          ("Two-tailed (τ ≠ 0)", "One-tailed (positive, τ > 0)", "One-tailed (negative, τ < 0)"), 
                          key="tail_tau_radio")

        st.info("This tab uses the Normal Approximation. For very small n (e.g., <10), exact tables for Kendall's Tau are preferred. `scipy.stats.kendalltau` can provide more precise p-values, especially for smaller n with ties.")

        z_calc_tau = float('nan')
        var_s = float('nan') # Variance of S (number of concordant - discordant pairs)
        
        if n_tau >= 2: # Basic condition for variance calculation
            var_s = (n_tau * (n_tau - 1) * (2 * n_tau + 5)) / 18
            if var_s > 0:
                se_tau_approx = math.sqrt(var_s) / (0.5 * n_tau * (n_tau - 1)) # SE for tau_a
                # A common approximation for z from tau_a (without continuity correction for simplicity here)
                # Tau_b is more complex with ties. This is a simplification.
                # z = tau / SE_tau ; SE_tau = sqrt( (2*(2n+5)) / (9n(n-1)) ) for tau_a
                # Using the SE for S and then standardizing S is another way: S = tau * n(n-1)/2
                # Z = S / sqrt(Var(S)) or (S +/- 1) / sqrt(Var(S)) with continuity correction
                # For simplicity, we'll use the direct z formula for tau as an approximation
                denominator_se_tau = (9 * n_tau * (n_tau - 1))
                if denominator_se_tau > 0:
                    se_tau_direct_approx = math.sqrt((2 * (2 * n_tau + 5)) / denominator_se_tau)
                    if se_tau_direct_approx > 0:
                         z_calc_tau = test_stat_tau / se_tau_direct_approx
                    elif test_stat_tau == 0:
                        z_calc_tau = 0.0
                    else:
                        z_calc_tau = float('inf') * np.sign(test_stat_tau)
                elif test_stat_tau == 0:
                     z_calc_tau = 0.0
                else:
                     z_calc_tau = float('inf') * np.sign(test_stat_tau)


        st.markdown(f"**Approximate z-statistic (from τ):** {format_value_for_display(z_calc_tau)}")

        st.subheader("Normal Distribution Plot (Approximation for τ)")
        fig_tau, ax_tau = plt.subplots(figsize=(8,5))
        
        crit_z_upper_tau_plot, crit_z_lower_tau_plot = None, None
        if tail_tau == "Two-tailed (τ ≠ 0)": 
            crit_z_upper_tau_plot = stats.norm.ppf(1 - alpha_tau / 2)
            crit_z_lower_tau_plot = stats.norm.ppf(alpha_tau / 2)
        elif tail_tau == "One-tailed (positive, τ > 0)": 
            crit_z_upper_tau_plot = stats.norm.ppf(1 - alpha_tau)
        else: # One-tailed (negative, τ < 0)
            crit_z_lower_tau_plot = stats.norm.ppf(alpha_tau)

        plot_min_z_tau = min(stats.norm.ppf(0.00000001), z_calc_tau - 2 if not np.isnan(z_calc_tau) else -2, -4.0)
        plot_max_z_tau = max(stats.norm.ppf(0.99999999), z_calc_tau + 2 if not np.isnan(z_calc_tau) else 2, 4.0)
        if not np.isnan(z_calc_tau) and abs(z_calc_tau) > 3.5:
            plot_min_z_tau = min(plot_min_z_tau, z_calc_tau - 0.5)
            plot_max_z_tau = max(plot_max_z_tau, z_calc_tau + 0.5)
        
        x_norm_tau_plot = np.linspace(plot_min_z_tau, plot_max_z_tau, 500)
        y_norm_tau_plot = stats.norm.pdf(x_norm_tau_plot)
        ax_tau.plot(x_norm_tau_plot, y_norm_tau_plot, 'b-', lw=2, label='Standard Normal Distribution (z)')

        if crit_z_upper_tau_plot is not None and not np.isnan(crit_z_upper_tau_plot):
            x_fill_upper = np.linspace(crit_z_upper_tau_plot, plot_max_z_tau, 100)
            ax_tau.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, label=f'Crit. Region (α={alpha_tau/(2 if tail_tau == "Two-tailed (τ ≠ 0)" else 1):.8f})')
            ax_tau.axvline(crit_z_upper_tau_plot, color='red', linestyle='--', lw=1)
        if crit_z_lower_tau_plot is not None and not np.isnan(crit_z_lower_tau_plot) and tail_tau == "Two-tailed (τ ≠ 0)":
            x_fill_lower = np.linspace(plot_min_z_tau, crit_z_lower_tau_plot, 100)
            ax_tau.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.5)
            ax_tau.axvline(crit_z_lower_tau_plot, color='red', linestyle='--', lw=1)
        
        if not np.isnan(z_calc_tau):
            ax_tau.axvline(z_calc_tau, color='green', linestyle='-', lw=2, label=f'Approx. z_calc = {z_calc_tau:.3f}')
        ax_tau.set_title('Normal Approximation for Kendall\'s τ'); ax_tau.legend(); ax_tau.grid(True); st.pyplot(fig_tau)

        st.subheader("Standard Normal Table: Cumulative P(Z < z)")
        st.markdown("This table shows the area to the left of a given z-score. The z-critical value (derived from your alpha and tail selection) is used for highlighting.")
        
        z_crit_for_table_tau = crit_z_upper_tau_plot if tail_tau == "One-tailed (positive, τ > 0)" else \
                               (crit_z_upper_tau_plot if crit_z_upper_tau_plot is not None and tail_tau == "Two-tailed (τ ≠ 0)" else \
                               (crit_z_lower_tau_plot if crit_z_lower_tau_plot is not None else 0.0) )
        if z_crit_for_table_tau is None or np.isnan(z_crit_for_table_tau): z_crit_for_table_tau = 0.0

        all_z_row_labels_tau = [f"{val:.1f}" for val in np.round(np.arange(-3.4, 3.5, 0.1), 1)]
        z_col_labels_str_tau = [f"{val:.2f}" for val in np.round(np.arange(0.00, 0.10, 0.01), 2)]
        
        z_target_for_table_row_numeric_tau = round(z_crit_for_table_tau, 1) 

        try:
            closest_row_idx_tau = min(range(len(all_z_row_labels_tau)), key=lambda i: abs(float(all_z_row_labels_tau[i]) - z_target_for_table_row_numeric_tau))
        except ValueError: 
            closest_row_idx_tau = len(all_z_row_labels_tau) // 2

        window_size_z_tau = 5
        start_idx_z_tau = max(0, closest_row_idx_tau - window_size_z_tau)
        end_idx_z_tau = min(len(all_z_row_labels_tau), closest_row_idx_tau + window_size_z_tau + 1)
        z_table_display_rows_str_tau = all_z_row_labels_tau[start_idx_z_tau:end_idx_z_tau]

        table_data_z_lookup_tau = []
        for z_r_str_idx in z_table_display_rows_str_tau:
            z_r_val = float(z_r_str_idx)
            row = { 'z': z_r_str_idx } 
            for z_c_str_idx in z_col_labels_str_tau:
                z_c_val = float(z_c_str_idx)
                current_z_val = round(z_r_val + z_c_val, 2)
                prob = stats.norm.cdf(current_z_val)
                row[z_c_str_idx] = format_value_for_display(prob, decimals=4)
            table_data_z_lookup_tau.append(row)
        
        df_z_lookup_table_tau = pd.DataFrame(table_data_z_lookup_tau).set_index('z')
        
        # Use the generic z-table styling function (needs to be defined in the main script or passed)
        # For now, assuming a style_z_lookup_table function exists similar to the one in tab_z_distribution
        # This function would take df_to_style and the z_crit_val_to_highlight as arguments
        def style_z_lookup_table_generic(df_to_style, z_crit_val_to_highlight):
            data = df_to_style 
            style_df = pd.DataFrame('', index=data.index, columns=data.columns)
            try:
                z_target_base_numeric = round(z_crit_val_to_highlight,1) 
                actual_row_labels_float = [float(label) for label in data.index]
                closest_row_float_val = min(actual_row_labels_float, key=lambda x_val: abs(x_val - z_target_base_numeric))
                highlight_row_label = f"{closest_row_float_val:.1f}"

                z_target_second_decimal_part = round(abs(z_crit_val_to_highlight - closest_row_float_val), 2) 
                actual_col_labels_float = [float(col_str) for col_str in data.columns]
                closest_col_float_val = min(actual_col_labels_float, key=lambda x_val: abs(x_val - z_target_second_decimal_part))
                highlight_col_label = f"{closest_col_float_val:.2f}"

                if highlight_row_label in style_df.index:
                    for col_name_iter in style_df.columns: 
                        style_df.loc[highlight_row_label, col_name_iter] = 'background-color: lightblue;'
                if highlight_col_label in style_df.columns:
                    for r_idx_iter in style_df.index: 
                        current_style = style_df.loc[r_idx_iter, highlight_col_label]
                        style_df.loc[r_idx_iter, highlight_col_label] = (current_style + ';' if current_style and not current_style.endswith(';') else current_style) + 'background-color: lightgreen;'
                if highlight_row_label in style_df.index and highlight_col_label in style_df.columns:
                    current_cell_style = style_df.loc[highlight_row_label, highlight_col_label]
                    style_df.loc[highlight_row_label, highlight_col_label] = (current_cell_style + ';' if current_cell_style and not current_cell_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            except Exception: pass 
            return style_df

        st.markdown(df_z_lookup_table_tau.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                               {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_z_lookup_table_generic, z_crit_val_to_highlight=z_crit_for_table_tau, axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows P(Z < z). Highlighted cell, row, and column are closest to the z-critical value derived from α={alpha_tau:.8f} and tail selection.")


    with col2: 
        st.subheader("P-value Calculation Explanation (Normal Approximation)")
        p_val_calc_tau = float('nan')
        
        if not np.isnan(z_calc_tau):
            if tail_tau == "Two-tailed (τ ≠ 0)": p_val_calc_tau = 2 * stats.norm.sf(abs(z_calc_tau))
            elif tail_tau == "One-tailed (positive, τ > 0)": p_val_calc_tau = stats.norm.sf(z_calc_tau)
            else:  p_val_calc_tau = stats.norm.cdf(z_calc_tau) # One-tailed (negative)
            p_val_calc_tau = min(p_val_calc_tau, 1.0) if not np.isnan(p_val_calc_tau) else float('nan')
        
        st.markdown(f"""
        The Kendall's τ statistic ({test_stat_tau:.3f}) is converted to an approximate z-statistic ({format_value_for_display(z_calc_tau)}).
        The p-value is then derived from the standard normal distribution.
        * **Two-tailed**: `2 * P(Z ≥ |{format_value_for_display(z_calc_tau)}|)`
        * **One-tailed (positive, τ > 0)**: `P(Z ≥ {format_value_for_display(z_calc_tau)})` 
        * **One-tailed (negative, τ < 0)**: `P(Z ≤ {format_value_for_display(z_calc_tau)})` 
        """)

        st.subheader("Summary (Normal Approximation)")
        crit_val_z_display_tau = "N/A"
        if tail_tau == "Two-tailed (τ ≠ 0)": crit_val_z_display_tau = f"±{format_value_for_display(crit_z_upper_tau_plot)}"
        elif tail_tau == "One-tailed (positive, τ > 0)": crit_val_z_display_tau = format_value_for_display(crit_z_upper_tau_plot)
        else: crit_val_z_display_tau = format_value_for_display(crit_z_lower_tau_plot)
        
        decision_crit_tau = False
        comparison_crit_str_tau = "N/A"
        if not np.isnan(z_calc_tau):
            if tail_tau == "Two-tailed (τ ≠ 0)" and crit_z_upper_tau_plot is not None:
                decision_crit_tau = abs(z_calc_tau) > crit_z_upper_tau_plot
                comparison_crit_str_tau = f"|Approx. z_calc ({abs(z_calc_tau):.3f})| {' > ' if decision_crit_tau else ' ≤ '} z_crit ({format_value_for_display(crit_z_upper_tau_plot)})"
            elif tail_tau == "One-tailed (positive, τ > 0)" and crit_z_upper_tau_plot is not None:
                decision_crit_tau = z_calc_tau > crit_z_upper_tau_plot
                comparison_crit_str_tau = f"Approx. z_calc ({z_calc_tau:.3f}) {' > ' if decision_crit_tau else ' ≤ '} z_crit ({format_value_for_display(crit_z_upper_tau_plot)})"
            elif tail_tau == "One-tailed (negative, τ < 0)" and crit_z_lower_tau_plot is not None:
                decision_crit_tau = z_calc_tau < crit_z_lower_tau_plot
                comparison_crit_str_tau = f"Approx. z_calc ({z_calc_tau:.3f}) {' < ' if decision_crit_tau else ' ≥ '} z_crit ({format_value_for_display(crit_z_lower_tau_plot)})"
        
        decision_p_alpha_tau = p_val_calc_tau < alpha_tau if not np.isnan(p_val_calc_tau) else False
        
        st.markdown(f"""
        1.  **Critical z-value ({tail_tau})**: {crit_val_z_display_tau}
            * *Significance level (α)*: {alpha_tau:.8f}
        2.  **Calculated τ-statistic**: {test_stat_tau:.3f} (Approx. z-statistic: {format_value_for_display(z_calc_tau)})
            * *Calculated p-value (Normal Approx.)*: {format_value_for_display(p_val_calc_tau, decimals=4)} ({apa_p_value(p_val_calc_tau)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_tau else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_tau}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_tau else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_tau)} is {'less than' if decision_p_alpha_tau else 'not less than'} α ({alpha_tau:.8f}).
        5.  **APA 7 Style Report (based on Normal Approximation)**:
            A Kendall's tau-b correlation was run to determine the relationship between the two variables. The correlation was found to be {'' if decision_p_alpha_tau else 'not '}statistically significant, τ<sub>b</sub> = {test_stat_tau:.2f}, {apa_p_value(p_val_calc_tau)} (using normal approximation for n={n_tau}). The null hypothesis was {'rejected' if decision_p_alpha_tau else 'not rejected'} at α = {alpha_tau:.2f}.
        """)

# --- Tab 14: Hartley's F_max Table ---
def tab_hartleys_fmax():
    st.header("Hartley's F_max Test for Homogeneity of Variances")
    st.markdown("Tests if the variances of several groups are equal. Assumes normality and equal sample sizes (though can be adapted for slightly unequal n).")
    
    col1, col2 = st.columns([2, 1.5])

    # Placeholder for F_max critical values: (k, df) -> critical_value
    # This is a very small subset for demonstration. A full table would be extensive.
    # Rows are df (n-1 per group), Columns are k (number of groups)
    F_MAX_TABLE_ALPHA_05 = { 
        # df: [k=2,  k=3,   k=4,   k=5,   k=6,   k=7,   k=8,   k=9,   k=10,  k=11,  k=12]
        2:  [39.0, 87.5,  142,   202,   266,   333,   403,   475,   550,   626,   704 ],
        3:  [15.4, 27.8,  39.2,  50.7,  62.0,  72.9,  83.5,  93.9,  104,   114,   124 ],
        4:  [9.60, 15.5,  20.6,  25.2,  29.5,  33.6,  37.5,  41.1,  44.6,  48.0,  51.4],
        5:  [7.15, 10.8,  13.7,  16.3,  18.7,  20.8,  22.9,  24.7,  26.5,  28.2,  29.9],
        10: [4.03, 5.34,  6.31,  7.11,  7.80,  8.41,  8.95,  9.45,  9.91,  10.3,  10.7],
        20: [2.95, 3.54,  4.01,  4.37,  4.68,  4.96,  5.20,  5.43,  5.63,  5.81,  5.98],
        30: [2.53, 2.92,  3.25,  3.50,  3.70,  3.87,  4.02,  4.16,  4.29,  4.40,  4.51],
        60: [2.00, 2.21,  2.39,  2.52,  2.63,  2.72,  2.80,  2.87,  2.94,  3.00,  3.05]
    }
    F_MAX_TABLE_ALPHA_01 = {
        # df: [k=2,   k=3,   k=4,   k=5,   k=6,   k=7,   k=8,   k=9,   k=10,  k=11,  k=12]
        2:  [199,  448,   729,  1036,  1362,  1699,  2047,  2404,  2770,  3142,  3521 ],
        3:  [47.5, 85,    120,   151,   184,   216,   249,   277,   310,   341,   370 ],
        4:  [23.2, 37,    49,    59,    69,    79,    89,    97,    106,   113,   120 ],
        5:  [14.9, 22,    28,    33,    38,    42,    46,    50,    54,    57,    60 ],
        10: [7.10, 9.47,  11.1,  12.5,  13.7,  14.8,  15.8,  16.7,  17.5,  18.3,  19.0 ],
        20: [4.45, 5.45,  6.16,  6.73,  7.20,  7.60,  7.95,  8.26,  8.53,  8.78,  9.02 ],
        30: [3.58, 4.20,  4.67,  5.03,  5.32,  5.57,  5.78,  5.97,  6.13,  6.28,  6.42 ],
        60: [2.66, 2.96,  3.19,  3.36,  3.51,  3.63,  3.73,  3.82,  3.90,  3.98,  4.04 ]
    }

    def get_fmax_critical_value(k_groups, df_per_group, alpha_level, table_05, table_01):
        table_to_use = table_05 if alpha_level == 0.05 else (table_01 if alpha_level == 0.01 else None)
        if table_to_use is None:
            return float('nan'), "Selected alpha not in embedded table (only 0.05 and 0.01 available)"

        available_dfs = sorted(table_to_use.keys())
        if not available_dfs: return float('nan'), "Embedded table empty"
        
        # Find closest df in table (use nearest lower if exact not found, or smallest if df_per_group is too small)
        closest_df = available_dfs[0] # Default to smallest df
        for current_df_in_table in available_dfs:
            if current_df_in_table <= df_per_group:
                closest_df = current_df_in_table
            else:
                break # Found the first df in table larger than df_per_group
        
        k_cols_in_table_header = [2,3,4,5,6,7,8,9,10,11,12] # k values for which we have data
        if k_groups not in k_cols_in_table_header:
            return float('nan'), f"k={k_groups} not in embedded table columns ({k_cols_in_table_header})"
        
        k_index = k_cols_in_table_header.index(k_groups) # This is the 0-based index for the list of values

        try:
            crit_val = table_to_use[closest_df][k_index]
            source_note = f"Used df={closest_df} from table for input df={df_per_group}." if closest_df != df_per_group else ""
            return crit_val, f"Embedded table (α={alpha_level}). {source_note}"
        except (KeyError, IndexError):
            return float('nan'), "Could not find value in embedded table for this k/df combination."


    with col1:
        st.subheader("Inputs")
        alpha_fmax = st.selectbox("Alpha (α)", [0.05, 0.01], index=0, key="alpha_fmax_input")
        k_fmax = st.number_input("Number of Groups (k)", min_value=2, max_value=12, value=3, step=1, key="k_fmax_input") # Limited by placeholder table
        n_per_group_fmax = st.number_input("Sample Size per Group (n)", min_value=3, value=10, step=1, key="n_per_group_fmax_input") 
        df_fmax = n_per_group_fmax - 1
        
        st.markdown(f"**Degrees of Freedom (df) for each group** = n - 1 = **{df_fmax}**")
        
        s_sq_largest = st.number_input("Largest Sample Variance (s²_max)", value=5.0, format="%.3f", min_value=0.001)
        s_sq_smallest = st.number_input("Smallest Sample Variance (s²_min)", value=1.0, format="%.3f", min_value=0.001)

        f_max_observed = float('nan')
        if s_sq_smallest > 0:
            f_max_observed = s_sq_largest / s_sq_smallest
        st.markdown(f"**Calculated F_max statistic:** {format_value_for_display(f_max_observed)}")

        st.warning("The critical value table embedded here is a **limited placeholder**. For accurate results, consult comprehensive Hartley's F_max tables or use alternative tests like Levene's or Bartlett's test (available in `scipy.stats`).")
        
        st.subheader(f"Hartley's F_max Critical Values (Illustrative Table for α={alpha_fmax})")
        current_fmax_table_data = F_MAX_TABLE_ALPHA_05 if alpha_fmax == 0.05 else F_MAX_TABLE_ALPHA_01
        
        df_fmax_table_display_rows = get_dynamic_df_window(sorted(current_fmax_table_data.keys()), df_fmax, window_size=3)

        fmax_display_data = []
        k_cols_in_table_fmax_header = [2,3,4,5,6,7,8,9,10,11,12] 

        for df_val in df_fmax_table_display_rows:
            row_dict = {'df (n-1)': df_val}
            if df_val in current_fmax_table_data:
                for i, k_val_table in enumerate(k_cols_in_table_fmax_header):
                    try:
                        row_dict[f'k={k_val_table}'] = current_fmax_table_data[df_val][i]
                    except IndexError:
                        row_dict[f'k={k_val_table}'] = "N/A"
            else: 
                 for k_val_table in k_cols_in_table_fmax_header:
                    row_dict[f'k={k_val_table}'] = "N/A (df not in table)"
            fmax_display_data.append(row_dict)

        df_fmax_display = pd.DataFrame(fmax_display_data).set_index('df (n-1)')
        
        def style_fmax_table(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            # Find closest df in table for highlighting row
            available_dfs_in_table = [int(idx_str) for idx_str in df_to_style.index if idx_str.isdigit()]
            closest_df_highlight = min(available_dfs_in_table, key=lambda x:abs(x-df_fmax)) if available_dfs_in_table else -1
            
            selected_df_fmax_str = str(closest_df_highlight if closest_df_highlight != -1 else df_fmax)


            if selected_df_fmax_str in df_to_style.index:
                style.loc[selected_df_fmax_str, :] = 'background-color: lightblue;'
            
            selected_k_fmax_col = f"k={k_fmax}"
            if selected_k_fmax_col in df_to_style.columns:
                style.loc[:, selected_k_fmax_col] = style.loc[:, selected_k_fmax_col].astype(str) + '; background-color: lightgreen;'
            if selected_df_fmax_str in df_to_style.index and selected_k_fmax_col in df_to_style.columns:
                current_style = style.loc[selected_df_fmax_str, selected_k_fmax_col]
                style.loc[selected_df_fmax_str, selected_k_fmax_col] = (current_style + ';' if current_style and not current_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style

        if not df_fmax_display.empty:
            st.markdown(df_fmax_display.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                          {'selector': 'td', 'props': [('text-align', 'center')]}])
                                         .apply(style_fmax_table, axis=None).to_html(), unsafe_allow_html=True)
        else:
            st.warning("Could not display F_max table based on inputs.")

    with col2:
        st.subheader("Summary")
        f_max_critical, f_max_source_note = get_fmax_critical_value(k_fmax, df_fmax, alpha_fmax, F_MAX_TABLE_ALPHA_05, F_MAX_TABLE_ALPHA_01)

        st.markdown(f"**Critical F_max (α={alpha_fmax}, k={k_fmax}, df={df_fmax})**: {format_value_for_display(f_max_critical)}")
        st.caption(f_max_source_note)

        decision_fmax = "N/A"
        if not np.isnan(f_max_observed) and not np.isnan(f_max_critical):
            if f_max_observed > f_max_critical:
                decision_fmax = "Reject H₀ (variances are likely unequal)"
            else:
                decision_fmax = "Fail to reject H₀ (no significant evidence of unequal variances)"
        
        st.markdown(f"""
        1.  **Observed F_max**: {format_value_for_display(f_max_observed)}
        2.  **Critical F_max**: {format_value_for_display(f_max_critical)}
        3.  **Decision**: {decision_fmax}
        """)
        st.markdown("""
        **Interpretation**: If Observed F_max > Critical F_max, reject the null hypothesis of equal variances.
        This test is sensitive to departures from normality. Consider Levene's test or Bartlett's test as alternatives for testing homogeneity of variances.
        """)

# --- Tab 15: Scheffé's Test Critical Values ---
def tab_scheffe():
    st.header("Scheffé's Test Critical Value Calculator")
    st.markdown("""
    Scheffé's test is a post-hoc test used in ANOVA to compare all possible simple and complex contrasts among group means 
    while maintaining a familywise error rate at the specified alpha level. The critical value is derived from the F-distribution.
    """)
    
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_scheffe = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_scheffe_input")
        k_scheffe = st.number_input("Number of Groups (k)", min_value=2, value=3, step=1, key="k_scheffe_input")
        df_error_scheffe = st.number_input("Degrees of Freedom for Error (df_error / df_within)", min_value=1, value=20, step=1, key="df_error_scheffe_input")
        
        f_observed_contrast = st.number_input("Calculated F-statistic for your contrast (optional, for decision)", value=0.0, format="%.3f", min_value=0.0, key="f_observed_contrast_scheffe")

        df_num_scheffe = k_scheffe - 1
        if df_num_scheffe <= 0:
            st.error("Number of groups (k) must be at least 2.")
            st.stop()

        st.markdown(f"**Numerator df (df₁)** = k - 1 = **{df_num_scheffe}**")
        st.markdown(f"**Denominator df (df₂)** = df_error = **{df_error_scheffe}**")

        f_crit_anova = float('nan')
        scheffe_crit_value = float('nan')
        if df_num_scheffe > 0 and df_error_scheffe > 0:
            try:
                f_crit_anova = stats.f.ppf(1 - alpha_scheffe, df_num_scheffe, df_error_scheffe)
                if not np.isnan(f_crit_anova):
                    scheffe_crit_value = (k_scheffe - 1) * f_crit_anova
            except Exception as e:
                st.warning(f"Could not calculate F-critical for Scheffe: {e}")
        
        st.markdown(f"**F-critical (for overall ANOVA at α={alpha_scheffe:.8f}, df₁={df_num_scheffe}, df₂={df_error_scheffe})**: {format_value_for_display(f_crit_anova)}")
        st.markdown(f"### **Scheffé Critical Value (S)** = (k-1) * F_crit = **{format_value_for_display(scheffe_crit_value)}**")


        st.subheader(f"F-Distribution Plot (df₁={df_num_scheffe}, df₂={df_error_scheffe})")
        fig_scheffe, ax_scheffe = plt.subplots(figsize=(8,5))
        
        if df_num_scheffe > 0 and df_error_scheffe > 0:
            plot_max_f_scheffe = max(stats.f.ppf(0.999, df_num_scheffe, df_error_scheffe) if df_num_scheffe > 0 and df_error_scheffe > 0 else 10.0, 
                                     f_observed_contrast * 1.5 if f_observed_contrast > 0 else 5.0, 
                                     scheffe_crit_value * 1.1 if not np.isnan(scheffe_crit_value) else 5.0,
                                     f_crit_anova * 1.5 if not np.isnan(f_crit_anova) else 5.0)
            
            x_f_scheffe_plot = np.linspace(0.001, plot_max_f_scheffe, 500)
            y_f_scheffe_plot = stats.f.pdf(x_f_scheffe_plot, df_num_scheffe, df_error_scheffe)
            ax_scheffe.plot(x_f_scheffe_plot, y_f_scheffe_plot, 'b-', lw=2, label=f'F-dist (df₁={df_num_scheffe}, df₂={df_error_scheffe})')

            if not np.isnan(f_crit_anova):
                ax_scheffe.axvline(f_crit_anova, color='purple', linestyle=':', lw=1.5, label=f'F_crit(ANOVA) = {f_crit_anova:.3f}')

            if not np.isnan(scheffe_crit_value):
                ax_scheffe.axvline(scheffe_crit_value, color='red', linestyle='--', lw=2, label=f'Scheffé Crit (S) = {scheffe_crit_value:.3f}')
                # Shading for Scheffe's critical value
                x_fill_scheffe_crit_region = np.linspace(scheffe_crit_value, plot_max_f_scheffe, 100)
                if len(x_fill_scheffe_crit_region) > 1 : # Ensure there's a region to fill
                    ax_scheffe.fill_between(x_fill_scheffe_crit_region, stats.f.pdf(x_fill_scheffe_crit_region, df_num_scheffe, df_error_scheffe), color='lightcoral', alpha=0.3, label=f'Scheffé Rejection Region')


            if f_observed_contrast > 0:
                 ax_scheffe.axvline(f_observed_contrast, color='green', linestyle='-', lw=2, label=f'F_obs(contrast) = {f_observed_contrast:.3f}')
        else:
            ax_scheffe.text(0.5,0.5, "df must be > 0 for plot.", ha='center')

        ax_scheffe.set_title(f'F-Distribution and Scheffé Critical Value')
        ax_scheffe.set_xlabel('F-value')
        ax_scheffe.set_ylabel('Probability Density')
        ax_scheffe.legend(); ax_scheffe.grid(True); st.pyplot(fig_scheffe)

        st.subheader(f"Scheffé Critical Values (S) for α = {alpha_scheffe:.8f}")
        all_df_error_options_scheffe = list(range(1, 31)) + [35,40,45,50,60,70,80,90,100,120, 200, 500, 1000]
        table_df_error_window_scheffe = get_dynamic_df_window(all_df_error_options_scheffe, df_error_scheffe, window_size=3)
        table_k_cols_scheffe = [2, 3, 4, 5, 6, 8, 10] 

        scheffe_table_data = []
        for df_err_iter in table_df_error_window_scheffe:
            df_err_calc = int(df_err_iter)
            if df_err_calc <=0: continue
            row = {'df_error': str(df_err_calc)}
            for k_val_iter in table_k_cols_scheffe:
                df_num_iter = k_val_iter - 1
                if df_num_iter <=0: 
                    row[f"k={k_val_iter}"] = "N/A"
                    continue
                f_c_cell = float('nan')
                s_c_cell = float('nan')
                try:
                    f_c_cell = stats.f.ppf(1 - alpha_scheffe, df_num_iter, df_err_calc)
                    if not np.isnan(f_c_cell):
                        s_c_cell = (k_val_iter - 1) * f_c_cell
                except Exception: pass # Keep as NaN if error
                row[f"k={k_val_iter}"] = format_value_for_display(s_c_cell)
            scheffe_table_data.append(row)
        
        df_scheffe_table = pd.DataFrame(scheffe_table_data).set_index('df_error')

        def style_scheffe_table(df_to_style): 
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_err_str = str(df_error_scheffe)
            
            if selected_df_err_str in df_to_style.index:
                style.loc[selected_df_err_str, :] = 'background-color: lightblue;'
            
            closest_k_col_val_s = min(table_k_cols_scheffe, key=lambda x: abs(x - k_scheffe))
            highlight_col_name_s = f"k={closest_k_col_val_s}"

            if highlight_col_name_s in df_to_style.columns:
                for r_idx in df_to_style.index:
                    current_r_style = style.loc[r_idx, highlight_col_name_s]
                    style.loc[r_idx, highlight_col_name_s] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                if selected_df_err_str in df_to_style.index : 
                    current_c_style = style.loc[selected_df_err_str, highlight_col_name_s]
                    style.loc[selected_df_err_str, highlight_col_name_s] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style

        if not df_scheffe_table.empty:
            st.markdown(df_scheffe_table.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                           {'selector': 'td', 'props': [('text-align', 'center')]}])
                                         .apply(style_scheffe_table, axis=None).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows Scheffé critical values (S) for α={alpha_scheffe:.8f}. Highlighted for df_error closest to {df_error_scheffe} and k closest to {k_scheffe}.")
        else:
            st.warning("Scheffé critical value table could not be generated for current inputs.")

    with col2:
        st.subheader("Summary for Scheffé's Test")
        st.markdown(f"""
        Scheffé's test is used for any and all post-hoc comparisons (simple or complex) after a significant ANOVA.
        A contrast is significant if its calculated F-statistic (F<sub>contrast</sub>) exceeds the Scheffé critical value (S).
        
        **S = (k - 1) * F<sub>α, (k-1), df_error</sub>**
        """)
        
        decision_scheffe = "N/A"
        comparison_str_scheffe = "F_obs(contrast) not provided or S not calculable."
        if f_observed_contrast > 0 and not np.isnan(scheffe_crit_value):
            if f_observed_contrast > scheffe_crit_value:
                decision_scheffe = "Reject H₀ for this contrast (contrast is significant)"
            else:
                decision_scheffe = "Fail to reject H₀ for this contrast (contrast is not significant)"
            comparison_str_scheffe = f"F_obs(contrast) ({f_observed_contrast:.3f}) {' > ' if f_observed_contrast > scheffe_crit_value else ' ≤ '} S ({format_value_for_display(scheffe_crit_value)})"

        # P-value for Scheffe's test (for a specific contrast)
        # The p-value isn't directly from the Scheffe critical value S.
        # Instead, you compare F_observed_contrast to S.
        # If you wanted a p-value *associated with the Scheffe criterion*, it would be complex.
        # Usually, the decision is based on F_obs vs S.
        # For reporting, the p-value of the *original contrast F-statistic* might be reported,
        # with the note that Scheffe's criterion was used for significance.
        p_val_contrast_f = float('nan')
        if df_num_scheffe > 0 and df_error_scheffe > 0 and f_observed_contrast > 0:
            try:
                # This is the p-value of the F-statistic for the contrast itself, not adjusted by Scheffe
                # It's provided for context; the decision is based on F_obs vs S.
                p_val_contrast_f = stats.f.sf(f_observed_contrast, df_num_scheffe, df_error_scheffe)
            except Exception:
                pass


        st.markdown(f"""
        1.  **Number of Groups (k)**: {k_scheffe}
        2.  **Degrees of Freedom for Error (df_error)**: {df_error_scheffe}
        3.  **Significance Level (α)**: {alpha_scheffe:.8f}
        4.  **F-critical for overall ANOVA (df₁={df_num_scheffe}, df₂={df_error_scheffe})**: {format_value_for_display(f_crit_anova)}
        5.  **Scheffé Critical Value (S)**: **{format_value_for_display(scheffe_crit_value)}**
        6.  **Your Calculated F for a specific contrast (F<sub>contrast</sub>)**: {f_observed_contrast:.3f} (p-value for this F: {apa_p_value(p_val_contrast_f)})
        7.  **Decision for your contrast (using Scheffé criterion)**: {decision_scheffe}
            * *Reason*: {comparison_str_scheffe}
        """)
        st.markdown("""
        **APA Style Example (for a specific contrast):**
        A Scheffé post-hoc test was used to evaluate the significance of the contrast. The observed F-statistic for the contrast was F(1, {df_error_scheffe}) = {f_observed_contrast:.2f}. This was {'' if (f_observed_contrast > scheffe_crit_value and not np.isnan(scheffe_crit_value)) else 'not '}greater than the Scheffé critical value of S = {format_value_for_display(scheffe_crit_value, decimals=2)}, indicating the contrast was {'' if (f_observed_contrast > scheffe_crit_value and not np.isnan(scheffe_crit_value)) else 'not '}statistically significant at the α = {alpha_scheffe:.2f} level.
        *(Adjust based on your actual contrast and significance)*.
        """)

# Helper function for z-table styling (used by Mann-Whitney, Wilcoxon, Tukey Approx)
def style_z_lookup_table(df_to_style, z_crit_val_to_highlight):
    data = df_to_style 
    style_df = pd.DataFrame('', index=data.index, columns=data.columns)
    try:
        z_target_base_numeric = round(z_crit_val_to_highlight,1) 
        actual_row_labels_float = [float(label) for label in data.index]
        closest_row_float_val = min(actual_row_labels_float, key=lambda x_val: abs(x_val - z_target_base_numeric))
        highlight_row_label = f"{closest_row_float_val:.1f}"

        z_target_second_decimal_part = round(abs(z_crit_val_to_highlight - closest_row_float_val), 2) 
        actual_col_labels_float = [float(col_str) for col_str in data.columns]
        closest_col_float_val = min(actual_col_labels_float, key=lambda x_val: abs(x_val - z_target_second_decimal_part))
        highlight_col_label = f"{closest_col_float_val:.2f}"

        if highlight_row_label in style_df.index:
            for col_name_iter in style_df.columns: 
                style_df.loc[highlight_row_label, col_name_iter] = 'background-color: lightblue;'
        if highlight_col_label in style_df.columns:
            for r_idx_iter in style_df.index: 
                current_style = style_df.loc[r_idx_iter, highlight_col_label]
                style_df.loc[r_idx_iter, highlight_col_label] = (current_style + ';' if current_style and not current_style.endswith(';') else current_style) + 'background-color: lightgreen;'
        if highlight_row_label in style_df.index and highlight_col_label in style_df.columns:
            current_cell_style = style_df.loc[highlight_row_label, highlight_col_label]
            style_df.loc[highlight_row_label, highlight_col_label] = (current_cell_style + ';' if current_cell_style and not current_cell_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
    except Exception: pass 
    return style_df

# --- Tab 13: Kendall's Tau Critical Values Table ---
def tab_kendalls_tau():
    st.header("Kendall's Tau (τ) Significance Test (Normal Approximation)")
    st.markdown("Tests the significance of Kendall's rank correlation coefficient (tau). For n > 10, a normal approximation is often used.")
    
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_tau = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_tau_input")
        n_tau = st.number_input("Sample Size (n, number of pairs)", min_value=4, value=20, step=1, key="n_tau_input") # Normal approx usually for n > 10
        test_stat_tau = st.number_input("Your Calculated Kendall's τ (tau)", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, format="%.3f", key="test_stat_tau_input")
        tail_tau = st.radio("Tail Selection (H₁)", 
                          ("Two-tailed (τ ≠ 0)", "One-tailed (positive, τ > 0)", "One-tailed (negative, τ < 0)"), 
                          key="tail_tau_radio")

        st.info("This tab uses the Normal Approximation. For very small n, exact tables for Kendall's Tau are preferred.")

        # Calculate z-statistic for Kendall's Tau (approximation)
        # Formula for SE_tau: sqrt((2 * (2n + 5)) / (9n * (n - 1)))
        # For p-value, scipy.stats.kendalltau uses a more complex calculation for exact p for small N or normal approx with continuity correction.
        # Here, we'll use the basic normal approximation for the z_calc for demonstration.
        z_calc_tau = float('nan')
        if n_tau >= 4 : # Need n > 1 for SE formula
            try:
                se_tau = math.sqrt((2 * (2 * n_tau + 5)) / (9 * n_tau * (n_tau - 1)))
                if se_tau > 0:
                    z_calc_tau = test_stat_tau / se_tau
                elif test_stat_tau == 0:
                    z_calc_tau = 0.0
                else: # se_tau is 0 but tau is not, implies perfect correlation or issue
                    z_calc_tau = float('inf') * np.sign(test_stat_tau)

            except ZeroDivisionError: # Should be caught by n_tau >=4
                z_calc_tau = float('nan')
        
        st.markdown(f"**Approximate z-statistic (from τ):** {format_value_for_display(z_calc_tau)}")

        st.subheader("Normal Distribution Plot (Approximation for τ)")
        fig_tau, ax_tau = plt.subplots(figsize=(8,5))
        
        crit_z_upper_tau_plot, crit_z_lower_tau_plot = None, None
        if tail_tau == "Two-tailed (τ ≠ 0)": 
            crit_z_upper_tau_plot = stats.norm.ppf(1 - alpha_tau / 2)
            crit_z_lower_tau_plot = stats.norm.ppf(alpha_tau / 2)
        elif tail_tau == "One-tailed (positive, τ > 0)": 
            crit_z_upper_tau_plot = stats.norm.ppf(1 - alpha_tau)
        else: # One-tailed (negative, τ < 0)
            crit_z_lower_tau_plot = stats.norm.ppf(alpha_tau)

        plot_min_z_tau = min(stats.norm.ppf(0.00000001), z_calc_tau - 2 if not np.isnan(z_calc_tau) else -2, -4.0)
        plot_max_z_tau = max(stats.norm.ppf(0.99999999), z_calc_tau + 2 if not np.isnan(z_calc_tau) else 2, 4.0)
        if not np.isnan(z_calc_tau) and abs(z_calc_tau) > 3.5:
            plot_min_z_tau = min(plot_min_z_tau, z_calc_tau - 0.5)
            plot_max_z_tau = max(plot_max_z_tau, z_calc_tau + 0.5)
        
        x_norm_tau_plot = np.linspace(plot_min_z_tau, plot_max_z_tau, 500)
        y_norm_tau_plot = stats.norm.pdf(x_norm_tau_plot)
        ax_tau.plot(x_norm_tau_plot, y_norm_tau_plot, 'b-', lw=2, label='Standard Normal Distribution (z)')

        if crit_z_upper_tau_plot is not None and not np.isnan(crit_z_upper_tau_plot):
            x_fill_upper = np.linspace(crit_z_upper_tau_plot, plot_max_z_tau, 100)
            ax_tau.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, label=f'Crit. Region (α={alpha_tau/(2 if tail_tau == "Two-tailed (τ ≠ 0)" else 1):.8f})')
            ax_tau.axvline(crit_z_upper_tau_plot, color='red', linestyle='--', lw=1)
        if crit_z_lower_tau_plot is not None and not np.isnan(crit_z_lower_tau_plot) and tail_tau == "Two-tailed (τ ≠ 0)":
            x_fill_lower = np.linspace(plot_min_z_tau, crit_z_lower_tau_plot, 100)
            ax_tau.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.5)
            ax_tau.axvline(crit_z_lower_tau_plot, color='red', linestyle='--', lw=1)
        
        if not np.isnan(z_calc_tau):
            ax_tau.axvline(z_calc_tau, color='green', linestyle='-', lw=2, label=f'Approx. z_calc = {z_calc_tau:.3f}')
        ax_tau.set_title('Normal Approximation for Kendall\'s τ'); ax_tau.legend(); ax_tau.grid(True); st.pyplot(fig_tau)

        st.subheader("Standard Normal Table: Cumulative P(Z < z)")
        st.markdown("This table shows the area to the left of a given z-score. The z-critical value (derived from your alpha and tail selection) is used for highlighting.")
        
        z_crit_for_table_tau = crit_z_upper_tau_plot if tail_tau == "One-tailed (positive, τ > 0)" else \
                               (crit_z_upper_tau_plot if crit_z_upper_tau_plot is not None and tail_tau == "Two-tailed (τ ≠ 0)" else \
                               (crit_z_lower_tau_plot if crit_z_lower_tau_plot is not None else 0.0) )
        if z_crit_for_table_tau is None or np.isnan(z_crit_for_table_tau): z_crit_for_table_tau = 0.0

        all_z_row_labels_tau = [f"{val:.1f}" for val in np.round(np.arange(-3.4, 3.5, 0.1), 1)]
        z_col_labels_str_tau = [f"{val:.2f}" for val in np.round(np.arange(0.00, 0.10, 0.01), 2)]
        
        z_target_for_table_row_numeric_tau = round(z_crit_for_table_tau, 1) 

        try:
            closest_row_idx_tau = min(range(len(all_z_row_labels_tau)), key=lambda i: abs(float(all_z_row_labels_tau[i]) - z_target_for_table_row_numeric_tau))
        except ValueError: 
            closest_row_idx_tau = len(all_z_row_labels_tau) // 2

        window_size_z_tau = 5
        start_idx_z_tau = max(0, closest_row_idx_tau - window_size_z_tau)
        end_idx_z_tau = min(len(all_z_row_labels_tau), closest_row_idx_tau + window_size_z_tau + 1)
        z_table_display_rows_str_tau = all_z_row_labels_tau[start_idx_z_tau:end_idx_z_tau]

        table_data_z_lookup_tau = []
        for z_r_str_idx in z_table_display_rows_str_tau:
            z_r_val = float(z_r_str_idx)
            row = { 'z': z_r_str_idx } 
            for z_c_str_idx in z_col_labels_str_tau:
                z_c_val = float(z_c_str_idx)
                current_z_val = round(z_r_val + z_c_val, 2)
                prob = stats.norm.cdf(current_z_val)
                row[z_c_str_idx] = format_value_for_display(prob, decimals=4)
            table_data_z_lookup_tau.append(row)
        
        df_z_lookup_table_tau = pd.DataFrame(table_data_z_lookup_tau).set_index('z')
        
        # Use the generic z-table styling function
        st.markdown(df_z_lookup_table_tau.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                               {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(lambda df: style_z_lookup_table(df, z_crit_for_table_tau), axis=None).to_html(), unsafe_allow_html=True) # Pass z_crit_for_table_tau
        st.caption(f"Table shows P(Z < z). Highlighted cell, row, and column are closest to the z-critical value derived from α={alpha_tau:.8f} and tail selection.")


    with col2: 
        st.subheader("P-value Calculation Explanation (Normal Approximation)")
        p_val_calc_tau = float('nan')
        
        if not np.isnan(z_calc_tau):
            if tail_tau == "Two-tailed (τ ≠ 0)": p_val_calc_tau = 2 * stats.norm.sf(abs(z_calc_tau))
            elif tail_tau == "One-tailed (positive, τ > 0)": p_val_calc_tau = stats.norm.sf(z_calc_tau)
            else:  p_val_calc_tau = stats.norm.cdf(z_calc_tau) # One-tailed (negative)
            p_val_calc_tau = min(p_val_calc_tau, 1.0) if not np.isnan(p_val_calc_tau) else float('nan')
        
        st.markdown(f"""
        The Kendall's τ statistic ({test_stat_tau:.3f}) is converted to an approximate z-statistic ({format_value_for_display(z_calc_tau)}).
        The p-value is then derived from the standard normal distribution.
        * **Two-tailed**: `2 * P(Z ≥ |{format_value_for_display(z_calc_tau)}|)`
        * **One-tailed (positive, τ > 0)**: `P(Z ≥ {format_value_for_display(z_calc_tau)})` 
        * **One-tailed (negative, τ < 0)**: `P(Z ≤ {format_value_for_display(z_calc_tau)})` 
        """)

        st.subheader("Summary (Normal Approximation)")
        crit_val_z_display_tau = "N/A"
        if tail_tau == "Two-tailed (τ ≠ 0)": crit_val_z_display_tau = f"±{format_value_for_display(crit_z_upper_tau_plot)}"
        elif tail_tau == "One-tailed (positive, τ > 0)": crit_val_z_display_tau = format_value_for_display(crit_z_upper_tau_plot)
        else: crit_val_z_display_tau = format_value_for_display(crit_z_lower_tau_plot)
        
        decision_crit_tau = False
        comparison_crit_str_tau = "N/A"
        if not np.isnan(z_calc_tau):
            if tail_tau == "Two-tailed (τ ≠ 0)" and crit_z_upper_tau_plot is not None:
                decision_crit_tau = abs(z_calc_tau) > crit_z_upper_tau_plot
                comparison_crit_str_tau = f"|Approx. z_calc ({abs(z_calc_tau):.3f})| {' > ' if decision_crit_tau else ' ≤ '} z_crit ({format_value_for_display(crit_z_upper_tau_plot)})"
            elif tail_tau == "One-tailed (positive, τ > 0)" and crit_z_upper_tau_plot is not None:
                decision_crit_tau = z_calc_tau > crit_z_upper_tau_plot
                comparison_crit_str_tau = f"Approx. z_calc ({z_calc_tau:.3f}) {' > ' if decision_crit_tau else ' ≤ '} z_crit ({format_value_for_display(crit_z_upper_tau_plot)})"
            elif tail_tau == "One-tailed (negative, τ < 0)" and crit_z_lower_tau_plot is not None:
                decision_crit_tau = z_calc_tau < crit_z_lower_tau_plot
                comparison_crit_str_tau = f"Approx. z_calc ({z_calc_tau:.3f}) {' < ' if decision_crit_tau else ' ≥ '} z_crit ({format_value_for_display(crit_z_lower_tau_plot)})"
        
        decision_p_alpha_tau = p_val_calc_tau < alpha_tau if not np.isnan(p_val_calc_tau) else False
        
        st.markdown(f"""
        1.  **Critical z-value ({tail_tau})**: {crit_val_z_display_tau}
            * *Significance level (α)*: {alpha_tau:.8f}
        2.  **Calculated τ-statistic**: {test_stat_tau:.3f} (Approx. z-statistic: {format_value_for_display(z_calc_tau)})
            * *Calculated p-value (Normal Approx.)*: {format_value_for_display(p_val_calc_tau, decimals=4)} ({apa_p_value(p_val_calc_tau)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_tau else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_tau}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_tau else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_tau)} is {'less than' if decision_p_alpha_tau else 'not less than'} α ({alpha_tau:.8f}).
        5.  **APA 7 Style Report (based on Normal Approximation)**:
            A Kendall's tau-b correlation was run to determine the relationship between the two variables. The correlation was found to be {'' if decision_p_alpha_tau else 'not '}statistically significant, τ<sub>b</sub> = {test_stat_tau:.2f}, {apa_p_value(p_val_calc_tau)} (using normal approximation for n={n_tau}). The null hypothesis was {'rejected' if decision_p_alpha_tau else 'not rejected'} at α = {alpha_tau:.2f}.
        """)

# --- Tab 14: Hartley's F_max Table ---
def tab_hartleys_fmax():
    st.header("Hartley's F_max Test for Homogeneity of Variances")
    st.markdown("Tests if the variances of several groups are equal. Assumes normality and equal sample sizes (though can be adapted for slightly unequal n).")
    
    col1, col2 = st.columns([2, 1.5])

    # Placeholder for F_max critical values: (k, df) -> critical_value
    # This would need to be populated with actual table values.
    # For demonstration, a very small subset.
    # Rows are df (n-1 per group), Columns are k (number of groups)
    # Alpha = 0.05
    F_MAX_TABLE_ALPHA_05 = { 
        # k=2,  k=3,   k=4,   k=5,   k=6
        2: [39.0, 87.5, 142,   202,   266],   # df = 2
        3: [15.4, 27.8, 39.2,  50.7,  62.0],  # df = 3
        4: [9.60, 15.5, 20.6,  25.2,  29.5],  # df = 4
        5: [7.15, 10.8, 13.7,  16.3,  18.7],  # df = 5
        10: [4.03, 5.34, 6.31,  7.11,  7.80], # df = 10
        20: [2.95, 3.54, 4.01,  4.37,  4.68], # df = 20
        # ... more df values
    }
    # Alpha = 0.01
    F_MAX_TABLE_ALPHA_01 = {
        # k=2,   k=3,   k=4,   k=5,   k=6
        2: [199,  448,   729,  1036,  1362],  # df = 2
        3: [47.5, 85,    120,   151,   184],   # df = 3
        4: [23.2, 37,    49,    59,    69],    # df = 4
        5: [14.9, 22,    28,    33,    38],    # df = 5
        10: [7.10, 9.47, 11.1,  12.5,  13.7],  # df = 10
        20: [4.45, 5.45, 6.16,  6.73,  7.20],  # df = 20
    }


    def get_fmax_critical_value(k_groups, df_per_group, alpha_level, table_05, table_01):
        table_to_use = table_05 if alpha_level == 0.05 else (table_01 if alpha_level == 0.01 else None)
        if table_to_use is None:
            return float('nan'), "Selected alpha not in embedded table"

        # Find closest df in table
        available_dfs = sorted(table_to_use.keys())
        if not available_dfs: return float('nan'), "Embedded table empty"
        
        closest_df = min(available_dfs, key=lambda x: abs(x - df_per_group))
        
        # k is usually 1-indexed for columns in tables (k=2 is often first actual column)
        # Our table_k_cols_display is 0-indexed for list access
        k_cols_in_table = [2,3,4,5,6] # Example: k values for which we have data
        if k_groups not in k_cols_in_table:
            return float('nan'), f"k={k_groups} not in embedded table columns ({k_cols_in_table})"
        
        k_index = k_cols_in_table.index(k_groups)

        try:
            crit_val = table_to_use[closest_df][k_index]
            source_note = f"Used df={closest_df} from table for input df={df_per_group}." if closest_df != df_per_group else ""
            return crit_val, f"Embedded table (α={alpha_level}). {source_note}"
        except (KeyError, IndexError):
            return float('nan'), "Could not find value in embedded table."


    with col1:
        st.subheader("Inputs")
        alpha_fmax = st.selectbox("Alpha (α)", [0.05, 0.01], index=0, key="alpha_fmax_input")
        k_fmax = st.number_input("Number of Groups (k)", min_value=2, max_value=6, value=3, step=1, key="k_fmax_input") # Limited by placeholder table
        # Assuming equal n per group for simplicity of df calculation for this example
        n_per_group_fmax = st.number_input("Sample Size per Group (n)", min_value=3, value=10, step=1, key="n_per_group_fmax_input") 
        df_fmax = n_per_group_fmax - 1
        
        st.markdown(f"**Degrees of Freedom (df) for each group** = n - 1 = **{df_fmax}**")
        
        s_sq_largest = st.number_input("Largest Sample Variance (s²_max)", value=5.0, format="%.3f", min_value=0.001)
        s_sq_smallest = st.number_input("Smallest Sample Variance (s²_min)", value=1.0, format="%.3f", min_value=0.001)

        f_max_observed = float('nan')
        if s_sq_smallest > 0:
            f_max_observed = s_sq_largest / s_sq_smallest
        st.markdown(f"**Calculated F_max statistic:** {format_value_for_display(f_max_observed)}")

        st.warning("The critical value table embedded here is a very small placeholder. For accurate results, consult comprehensive Hartley's F_max tables or use alternative tests like Levene's or Bartlett's.")
        
        st.subheader("Hartley's F_max Critical Values (Placeholder Table)")
        # Display a portion of the selected alpha table
        current_fmax_table_data = F_MAX_TABLE_ALPHA_05 if alpha_fmax == 0.05 else F_MAX_TABLE_ALPHA_01
        
        df_fmax_table_display_rows = get_dynamic_df_window(sorted(current_fmax_table_data.keys()), df_fmax, window_size=3)

        fmax_display_data = []
        k_cols_in_table_fmax = [2,3,4,5,6] # k values in our placeholder table

        for df_val in df_fmax_table_display_rows:
            row_dict = {'df (n-1)': df_val}
            if df_val in current_fmax_table_data:
                for i, k_val_table in enumerate(k_cols_in_table_fmax):
                    try:
                        row_dict[f'k={k_val_table}'] = current_fmax_table_data[df_val][i]
                    except IndexError:
                        row_dict[f'k={k_val_table}'] = "N/A"
            else: # df_val not in table
                 for k_val_table in k_cols_in_table_fmax:
                    row_dict[f'k={k_val_table}'] = "N/A"
            fmax_display_data.append(row_dict)

        df_fmax_display = pd.DataFrame(fmax_display_data).set_index('df (n-1)')
        
        def style_fmax_table(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_fmax_str = str(df_fmax)
            selected_k_fmax_col = f"k={k_fmax}"

            if selected_df_fmax_str in df_to_style.index:
                style.loc[selected_df_fmax_str, :] = 'background-color: lightblue;'
            if selected_k_fmax_col in df_to_style.columns:
                style.loc[:, selected_k_fmax_col] = style.loc[:, selected_k_fmax_col].astype(str) + 'background-color: lightgreen;'
            if selected_df_fmax_str in df_to_style.index and selected_k_fmax_col in df_to_style.columns:
                style.loc[selected_df_fmax_str, selected_k_fmax_col] = 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style

        if not df_fmax_display.empty:
            st.markdown(df_fmax_display.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                          {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_fmax_table, axis=None).to_html(), unsafe_allow_html=True)
        else:
            st.warning("Could not display F_max table based on inputs.")

    with col2:
        st.subheader("Summary")
        f_max_critical, f_max_source_note = get_fmax_critical_value(k_fmax, df_fmax, alpha_fmax, F_MAX_TABLE_ALPHA_05, F_MAX_TABLE_ALPHA_01)

        st.markdown(f"**Critical F_max (α={alpha_fmax}, k={k_fmax}, df={df_fmax})**: {format_value_for_display(f_max_critical)}")
        st.caption(f_max_source_note)

        decision_fmax = "N/A"
        if not np.isnan(f_max_observed) and not np.isnan(f_max_critical):
            if f_max_observed > f_max_critical:
                decision_fmax = "Reject H₀ (variances are likely unequal)"
            else:
                decision_fmax = "Fail to reject H₀ (no significant evidence of unequal variances)"
        
        st.markdown(f"""
        1.  **Observed F_max**: {format_value_for_display(f_max_observed)}
        2.  **Critical F_max**: {format_value_for_display(f_max_critical)}
        3.  **Decision**: {decision_fmax}
        """)
        st.markdown("""
        **Interpretation**: If Observed F_max > Critical F_max, reject the null hypothesis of equal variances.
        This test is sensitive to departures from normality. Consider Levene's test or Bartlett's test as alternatives.
        """)
# --- Tab 15: Scheffé's Test Critical Values ---
def tab_scheffe():
    st.header("Scheffé's Test Critical Value Calculator")
    st.markdown("""
    Scheffé's test is a post-hoc test used in ANOVA to compare all possible simple and complex contrasts among group means 
    while maintaining a familywise error rate at the specified alpha level. The critical value is derived from the F-distribution.
    """)
    
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_scheffe = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_scheffe_input")
        k_scheffe = st.number_input("Number of Groups (k)", min_value=2, value=3, step=1, key="k_scheffe_input")
        df_error_scheffe = st.number_input("Degrees of Freedom for Error (df_error / df_within)", min_value=1, value=20, step=1, key="df_error_scheffe_input")
        
        # Optional: User can input their calculated F for a specific contrast
        f_observed_contrast = st.number_input("Calculated F-statistic for your contrast (optional)", value=0.0, format="%.3f", min_value=0.0, key="f_observed_contrast_scheffe")

        df_num_scheffe = k_scheffe - 1
        if df_num_scheffe <= 0:
            st.error("Number of groups (k) must be at least 2.")
            st.stop()

        st.markdown(f"**Numerator df (df₁)** = k - 1 = **{df_num_scheffe}**")
        st.markdown(f"**Denominator df (df₂)** = df_error = **{df_error_scheffe}**")

        # Calculate critical F for Scheffe
        f_crit_anova = float('nan')
        scheffe_crit_value = float('nan')
        if df_num_scheffe > 0 and df_error_scheffe > 0:
            try:
                f_crit_anova = stats.f.ppf(1 - alpha_scheffe, df_num_scheffe, df_error_scheffe)
                if not np.isnan(f_crit_anova):
                    scheffe_crit_value = (k_scheffe - 1) * f_crit_anova
            except Exception as e:
                st.warning(f"Could not calculate F-critical for Scheffe: {e}")
        
        st.markdown(f"**F-critical (for ANOVA at α={alpha_scheffe:.8f}, df₁={df_num_scheffe}, df₂={df_error_scheffe})**: {format_value_for_display(f_crit_anova)}")
        st.markdown(f"### **Scheffé Critical Value (S)** = (k-1) * F_crit = **{format_value_for_display(scheffe_crit_value)}**")


        st.subheader(f"F-Distribution Plot (df₁={df_num_scheffe}, df₂={df_error_scheffe})")
        fig_scheffe, ax_scheffe = plt.subplots(figsize=(8,5))
        
        if df_num_scheffe > 0 and df_error_scheffe > 0:
            plot_max_f_scheffe = max(stats.f.ppf(0.999, df_num_scheffe, df_error_scheffe) if df_num_scheffe > 0 and df_error_scheffe > 0 else 10.0, 
                                     f_observed_contrast * 1.5 if f_observed_contrast > 0 else 5.0, 
                                     scheffe_crit_value * 1.1 if not np.isnan(scheffe_crit_value) else 5.0,
                                     f_crit_anova * 1.5 if not np.isnan(f_crit_anova) else 5.0)
            
            x_f_scheffe_plot = np.linspace(0.001, plot_max_f_scheffe, 500)
            y_f_scheffe_plot = stats.f.pdf(x_f_scheffe_plot, df_num_scheffe, df_error_scheffe)
            ax_scheffe.plot(x_f_scheffe_plot, y_f_scheffe_plot, 'b-', lw=2, label=f'F-dist (df₁={df_num_scheffe}, df₂={df_error_scheffe})')

            if not np.isnan(f_crit_anova):
                ax_scheffe.axvline(f_crit_anova, color='purple', linestyle=':', lw=1.5, label=f'F_crit(ANOVA) = {f_crit_anova:.3f}')

            if not np.isnan(scheffe_crit_value):
                x_fill_scheffe = np.linspace(scheffe_crit_value, plot_max_f_scheffe, 100)
                # PDF for filling might not be directly intuitive for Scheffe's S on F-dist plot
                # Instead, just mark the Scheffe critical value
                ax_scheffe.axvline(scheffe_crit_value, color='red', linestyle='--', lw=2, label=f'Scheffé Crit (S) = {scheffe_crit_value:.3f}')
                # Shade region beyond Scheffe_crit_value if an F_observed_contrast is given
                if f_observed_contrast > 0 and f_observed_contrast > scheffe_crit_value :
                     x_fill_observed_scheffe = np.linspace(scheffe_crit_value, f_observed_contrast, 100)
                     ax_scheffe.fill_between(x_fill_observed_scheffe, stats.f.pdf(x_fill_observed_scheffe, df_num_scheffe, df_error_scheffe), color='lightcoral', alpha=0.5)


            if f_observed_contrast > 0:
                 ax_scheffe.axvline(f_observed_contrast, color='green', linestyle='-', lw=2, label=f'F_obs(contrast) = {f_observed_contrast:.3f}')
        else:
            ax_scheffe.text(0.5,0.5, "df must be > 0 for plot.", ha='center')

        ax_scheffe.set_title(f'F-Distribution and Scheffé Critical Value')
        ax_scheffe.set_xlabel('F-value')
        ax_scheffe.set_ylabel('Probability Density')
        ax_scheffe.legend(); ax_scheffe.grid(True); st.pyplot(fig_scheffe)

        st.subheader(f"Scheffé Critical Values (S) for α = {alpha_scheffe:.8f}")
        all_df_error_options_scheffe = list(range(1, 31)) + [35,40,45,50,60,70,80,90,100,120, 200, 500, 1000]
        table_df_error_window_scheffe = get_dynamic_df_window(all_df_error_options_scheffe, df_error_scheffe, window_size=3)
        table_k_cols_scheffe = [2, 3, 4, 5, 6, 8, 10] 

        scheffe_table_data = []
        for df_err_iter in table_df_error_window_scheffe:
            df_err_calc = int(df_err_iter)
            if df_err_calc <=0: continue
            row = {'df_error': str(df_err_calc)}
            for k_val_iter in table_k_cols_scheffe:
                df_num_iter = k_val_iter - 1
                if df_num_iter <=0: 
                    row[f"k={k_val_iter}"] = "N/A"
                    continue
                f_c_cell = stats.f.ppf(1 - alpha_scheffe, df_num_iter, df_err_calc)
                s_c_cell = (k_val_iter - 1) * f_c_cell if not np.isnan(f_c_cell) else float('nan')
                row[f"k={k_val_iter}"] = format_value_for_display(s_c_cell)
            scheffe_table_data.append(row)
        
        df_scheffe_table = pd.DataFrame(scheffe_table_data).set_index('df_error')

        def style_scheffe_table(df_to_style): # Similar to F table styling
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_err_str = str(df_error_scheffe)
            
            if selected_df_err_str in df_to_style.index:
                style.loc[selected_df_err_str, :] = 'background-color: lightblue;'
            
            closest_k_col_val_s = min(table_k_cols_scheffe, key=lambda x: abs(x - k_scheffe))
            highlight_col_name_s = f"k={closest_k_col_val_s}"

            if highlight_col_name_s in df_to_style.columns:
                for r_idx in df_to_style.index:
                    current_r_style = style.loc[r_idx, highlight_col_name_s]
                    style.loc[r_idx, highlight_col_name_s] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                if selected_df_err_str in df_to_style.index : 
                    current_c_style = style.loc[selected_df_err_str, highlight_col_name_s]
                    style.loc[selected_df_err_str, highlight_col_name_s] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style

        if not df_scheffe_table.empty:
            st.markdown(df_scheffe_table.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                           {'selector': 'td', 'props': [('text-align', 'center')]}])
                                         .apply(style_scheffe_table, axis=None).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows Scheffé critical values (S) for α={alpha_scheffe:.8f}. Highlighted for df_error closest to {df_error_scheffe} and k closest to {k_scheffe}.")
        else:
            st.warning("Scheffé critical value table could not be generated for current inputs.")

    with col2:
        st.subheader("Summary for Scheffé's Test")
        st.markdown(f"""
        Scheffé's test is used for any and all post-hoc comparisons (simple or complex) after a significant ANOVA.
        A contrast is significant if its calculated F-statistic (F<sub>contrast</sub>) exceeds the Scheffé critical value (S).
        
        **S = (k - 1) * F<sub>α, (k-1), df_error</sub>**
        """)
        
        decision_scheffe = "N/A"
        comparison_str_scheffe = "F_obs(contrast) not provided or S not calculable."
        if f_observed_contrast > 0 and not np.isnan(scheffe_crit_value):
            if f_observed_contrast > scheffe_crit_value:
                decision_scheffe = "Reject H₀ for this contrast (contrast is significant)"
            else:
                decision_scheffe = "Fail to reject H₀ for this contrast (contrast is not significant)"
            comparison_str_scheffe = f"F_obs(contrast) ({f_observed_contrast:.3f}) {' > ' if f_observed_contrast > scheffe_crit_value else ' ≤ '} S ({format_value_for_display(scheffe_crit_value)})"

        st.markdown(f"""
        1.  **Number of Groups (k)**: {k_scheffe}
        2.  **Degrees of Freedom for Error (df_error)**: {df_error_scheffe}
        3.  **Significance Level (α)**: {alpha_scheffe:.8f}
        4.  **F-critical for overall ANOVA (df₁={df_num_scheffe}, df₂={df_error_scheffe})**: {format_value_for_display(f_crit_anova)}
        5.  **Scheffé Critical Value (S)**: **{format_value_for_display(scheffe_crit_value)}**
        6.  **Your Calculated F for a specific contrast (F<sub>contrast</sub>)**: {f_observed_contrast:.3f} (if provided)
        7.  **Decision for your contrast**: {decision_scheffe}
            * *Reason*: {comparison_str_scheffe}
        """)
        st.markdown("""
        **APA Style Example (for a specific contrast):**
        A Scheffé post-hoc test revealed that the mean for Group A (M=X.XX) was statistically significantly different from the mean of Group B (M=Y.YY), F<sub>contrast</sub>(1, {df_error_scheffe}) = {f_observed_contrast:.2f}, p < {alpha_scheffe:.2f} (Scheffé criterion S = {format_value_for_display(scheffe_crit_value, decimals=2)}). 
        *(Adjust based on your actual contrast and significance)*.
        """)

# --- Tab 16: Dunnett’s Test Table ---
def tab_dunnetts_test():
    st.header("Dunnett's Test Critical Values (Illustrative)")
    st.markdown("""
    Dunnett's test is used for comparing multiple treatment groups against a single control group, 
    while controlling the familywise error rate. True critical values come from a specialized distribution 
    (related to the multivariate t-distribution). 
    **The table provided here is a very small, illustrative placeholder for demonstration purposes only.**
    For accurate research, consult comprehensive Dunnett's test tables or use statistical software 
    (e.g., `statsmodels` in Python if available, or R).
    """)
    
    col1, col2 = st.columns([2, 1.5])

    # Illustrative placeholder for Dunnett's critical values (one-sided, alpha=0.05)
    # Rows are df_error, Columns are k (total number of groups including control)
    DUNNETT_TABLE_ALPHA_05_ONE_SIDED = {
        # df_error: [k=2,  k=3,   k=4,   k=5,   k=6,   k=7,   k=8,   k=9,  k=10]
        5:        [2.02, 2.44,  2.71,  2.90,  3.06,  3.19,  3.30,  3.40,  3.49],
        10:       [1.81, 2.15,  2.36,  2.51,  2.63,  2.73,  2.81,  2.89,  2.95],
        20:       [1.72, 2.02,  2.20,  2.33,  2.43,  2.51,  2.58,  2.64,  2.70],
        30:       [1.70, 1.98,  2.15,  2.28,  2.38,  2.45,  2.52,  2.58,  2.63],
        60:       [1.67, 1.95,  2.12,  2.23,  2.33,  2.40,  2.46,  2.51,  2.56],
        120:      [1.66, 1.93,  2.09,  2.20,  2.29,  2.36,  2.42,  2.47,  2.52],
        # infinity: [1.645,1.91,  2.06,  2.17,  2.26,  2.33,  2.39,  2.44,  2.48] # approx z for large df
    }
    # Illustrative placeholder for Dunnett's critical values (two-sided, alpha=0.05)
    DUNNETT_TABLE_ALPHA_05_TWO_SIDED = {
        # df_error: [k=2,  k=3,   k=4,   k=5,   k=6,   k=7,   k=8,   k=9,  k=10]
        5:        [2.57, 3.03,  3.33,  3.56,  3.75,  3.90,  4.04,  4.16,  4.26],
        10:       [2.23, 2.57,  2.79,  2.96,  3.10,  3.21,  3.31,  3.39,  3.47],
        20:       [2.09, 2.39,  2.58,  2.72,  2.83,  2.92,  3.00,  3.07,  3.13],
        30:       [2.04, 2.33,  2.51,  2.64,  2.75,  2.83,  2.90,  2.97,  3.02],
        60:       [2.00, 2.27,  2.44,  2.57,  2.66,  2.74,  2.80,  2.86,  2.91],
        120:      [1.98, 2.24,  2.41,  2.53,  2.62,  2.69,  2.75,  2.80,  2.85],
        # infinity: [1.96, 2.21,  2.39,  2.50,  2.58,  2.65,  2.71,  2.76,  2.81]
    }


    def get_dunnett_critical_value(k_total_groups, df_err, alpha_level, tail_type, table_05_one, table_05_two):
        # This function will be very limited due to placeholder tables
        if alpha_level != 0.05: # Only 0.05 tables are placeholders
            return float('nan'), "Only α=0.05 illustrative table available."
        
        table_to_use = table_05_one if tail_type == "One-sided" else table_05_two
        
        available_dfs = sorted(table_to_use.keys())
        if not available_dfs: return float('nan'), "Embedded illustrative table empty"
        
        closest_df = available_dfs[0]
        for current_df_in_table in available_dfs:
            if current_df_in_table <= df_err:
                closest_df = current_df_in_table
            else:
                break
        
        k_cols_in_table_header = list(range(2, 11)) # k values in our placeholder table
        if k_total_groups not in k_cols_in_table_header:
            return float('nan'), f"k={k_total_groups} not in illustrative table columns ({k_cols_in_table_header})"
        
        k_index = k_cols_in_table_header.index(k_total_groups)

        try:
            crit_val = table_to_use[closest_df][k_index]
            source_note = f"Used df={closest_df} from illustrative table for input df={df_err}." if closest_df != df_err else ""
            return crit_val, f"Illustrative table (α={alpha_level}, {tail_type}). {source_note}"
        except (KeyError, IndexError):
            return float('nan'), "Could not find value in illustrative table."

    with col1:
        st.subheader("Inputs")
        alpha_dunnett = st.selectbox("Alpha (α)", [0.05, 0.01], index=0, key="alpha_dunnett_input", help="Illustrative table only supports 0.05 for now.")
        k_total_dunnett = st.number_input("Total Number of Groups (k, including control)", min_value=2, max_value=10, value=3, step=1, key="k_total_dunnett_input")
        df_error_dunnett = st.number_input("Degrees of Freedom for Error (df_error / df_within)", min_value=2, value=20, step=1, key="df_error_dunnett_input")
        tail_dunnett = st.radio("Tail Selection", ("One-sided", "Two-sided"), key="tail_dunnett_radio")
        
        test_stat_dunnett_d = st.number_input("Your Calculated Dunnett's d-statistic (optional)", value=0.0, format="%.3f", key="test_stat_dunnett_d_input")

        st.subheader(f"Illustrative Dunnett's Critical Values (d) for α={alpha_dunnett}, {tail_dunnett}")
        
        current_dunnett_table_data = DUNNETT_TABLE_ALPHA_05_ONE_SIDED if tail_dunnett == "One-sided" and alpha_dunnett == 0.05 else \
                                     (DUNNETT_TABLE_ALPHA_05_TWO_SIDED if tail_dunnett == "Two-sided" and alpha_dunnett == 0.05 else None)

        if current_dunnett_table_data:
            df_dunnett_table_display_rows = get_dynamic_df_window(sorted(current_dunnett_table_data.keys()), df_error_dunnett, window_size=3)
            k_cols_dunnett_header = list(range(2,11))

            dunnett_display_data = []
            for df_val in df_dunnett_table_display_rows:
                row_dict = {'df_error': df_val}
                if df_val in current_dunnett_table_data:
                    for i, k_val_table in enumerate(k_cols_dunnett_header):
                        try:
                            row_dict[f'k={k_val_table}'] = current_dunnett_table_data[df_val][i]
                        except IndexError:
                            row_dict[f'k={k_val_table}'] = "N/A"
                else:
                    for k_val_table in k_cols_dunnett_header:
                        row_dict[f'k={k_val_table}'] = "N/A"
                dunnett_display_data.append(row_dict)
            
            df_dunnett_display = pd.DataFrame(dunnett_display_data).set_index('df_error')

            def style_dunnett_table(df_to_style):
                style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
                available_dfs_in_table = [int(idx_str) for idx_str in df_to_style.index if isinstance(idx_str, (int,float)) or idx_str.isdigit()]
                closest_df_highlight = min(available_dfs_in_table, key=lambda x:abs(x-df_error_dunnett)) if available_dfs_in_table else -1
                selected_df_str = str(closest_df_highlight if closest_df_highlight != -1 else df_error_dunnett)
                
                selected_k_col = f"k={k_total_dunnett}"

                if selected_df_str in df_to_style.index:
                    style.loc[selected_df_str, :] = 'background-color: lightblue;'
                if selected_k_col in df_to_style.columns:
                     style.loc[:, selected_k_col] = style.loc[:, selected_k_col].astype(str) + '; background-color: lightgreen;' # Ensure string concatenation
                if selected_df_str in df_to_style.index and selected_k_col in df_to_style.columns:
                    current_style_val = style.loc[selected_df_str, selected_k_col]
                    style.loc[selected_df_str, selected_k_col] = (current_style_val + ';' if current_style_val and not current_style_val.endswith(';') else current_style_val) + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
                return style

            if not df_dunnett_display.empty:
                st.markdown(df_dunnett_display.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                              {'selector': 'td', 'props': [('text-align', 'center')]}])
                                             .apply(style_dunnett_table, axis=None).to_html(), unsafe_allow_html=True)
            else:
                st.warning("Illustrative Dunnett's table could not be displayed for current inputs (likely alpha not 0.05).")
        else:
            st.warning("Illustrative Dunnett's table only available for α=0.05 in this demo.")

    with col2:
        st.subheader("Summary (Based on Illustrative Table)")
        dunnett_critical, dunnett_source_note = get_dunnett_critical_value(k_total_dunnett, df_error_dunnett, alpha_dunnett, tail_dunnett, DUNNETT_TABLE_ALPHA_05_ONE_SIDED, DUNNETT_TABLE_ALPHA_05_TWO_SIDED)

        st.markdown(f"**Illustrative Critical Dunnett's d (α={alpha_dunnett}, k={k_total_dunnett}, df_error={df_error_dunnett}, {tail_dunnett})**: {format_value_for_display(dunnett_critical)}")
        st.caption(dunnett_source_note)

        decision_dunnett = "N/A (or provide observed d)"
        comparison_str_dunnett = "Compare your observed d to the critical d."
        if test_stat_dunnett_d > 0 and not np.isnan(dunnett_critical):
            if abs(test_stat_dunnett_d) > dunnett_critical : # Dunnett's is usually about magnitude
                decision_dunnett = "Reject H₀ (treatment differs from control)"
            else:
                decision_dunnett = "Fail to reject H₀ (no significant difference from control)"
            comparison_str_dunnett = f"|Observed d ({abs(test_stat_dunnett_d):.3f})| {' > ' if abs(test_stat_dunnett_d) > dunnett_critical else ' ≤ '} Critical d ({format_value_for_display(dunnett_critical)})"
        
        st.markdown(f"""
        1.  **Your Calculated Dunnett's d-statistic**: {test_stat_dunnett_d:.3f}
        2.  **Illustrative Critical Dunnett's d**: {format_value_for_display(dunnett_critical)}
        3.  **Decision**: {decision_dunnett}
            * *Reason*: {comparison_str_dunnett}
        """)
        st.markdown("""
        **Interpretation**: If your calculated |d| (absolute value for two-sided) or d (for one-sided, in the correct direction) exceeds the critical d from a comprehensive table, the difference between that treatment group and the control group is statistically significant.
        """)

# --- Tab 17: Newman-Keuls Test Table (Simplified Normal Approximation) ---
def tab_newman_keuls():
    st.header("Newman-Keuls Test (Simplified Normal Approximation)")
    st.markdown("""
    The Newman-Keuls (SNK) test is a stepwise multiple comparison procedure used after a significant ANOVA. 
    It uses critical values from the Studentized Range (q) distribution, where the specific q-value depends 
    on the number of means spanned by a comparison (`p`) and `df_error`.
    **This tab provides a highly simplified approximation using the standard normal (z) distribution 
    and does NOT implement the true stepwise nature or the Studentized Range distribution.**
    Your input 'Calculated Statistic' will be treated as a z-score.
    For accurate Newman-Keuls results, use statistical software that implements the Studentized Range distribution 
    (e.g., `statsmodels` if available, or R).
    """)
    
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs for Normal Approximation")
        alpha_snk = st.number_input("Alpha (α) for z-comparison", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_snk_approx_input")
        # These are for context, as true SNK depends on them for q. Here, they are for APA report.
        k_snk_total_groups = st.number_input("Total Number of Groups in ANOVA (k)", min_value=2, value=4, step=1, key="k_snk_total_groups_context") 
        df_error_snk = st.number_input("Degrees of Freedom for Error (df_error)", min_value=1, value=20, step=1, key="df_error_snk_context")
        p_means_spanned = st.number_input("Number of means spanned by current comparison (p)", min_value=2, max_value=k_snk_total_groups, value=2, step=1, key="p_means_spanned_snk")

        test_stat_snk_q_as_z = st.number_input("Your Calculated Statistic for this comparison (q or other, treated as z)", value=1.0, format="%.3f", key="test_stat_snk_q_as_z_input")
        
        # SNK is typically one-sided for the q-value (q_obs > q_crit)
        # but we are approximating with z, so a one-sided z critical value is appropriate
        st.markdown("Comparisons in SNK are typically against an upper-tail critical q-value. We'll use an upper-tail z-critical value for this approximation.")

        st.subheader("Standard Normal (z) Distribution Plot")
        fig_snk_approx, ax_snk_approx = plt.subplots(figsize=(8,5))
        
        z_crit_upper_snk_approx = stats.norm.ppf(1 - alpha_snk) # One-tailed critical z

        plot_min_z_snk = min(stats.norm.ppf(0.00000001), test_stat_snk_q_as_z - 2, -4.0)
        plot_max_z_snk = max(stats.norm.ppf(0.99999999), test_stat_snk_q_as_z + 2, 4.0)
        if abs(test_stat_snk_q_as_z) > 3.5 : 
             plot_min_z_snk = min(plot_min_z_snk, test_stat_snk_q_as_z - 0.5)
             plot_max_z_snk = max(plot_max_z_snk, test_stat_snk_q_as_z + 0.5)

        x_z_snk_plot = np.linspace(plot_min_z_snk, plot_max_z_snk, 500)
        y_z_snk_plot = stats.norm.pdf(x_z_snk_plot)
        ax_snk_approx.plot(x_z_snk_plot, y_z_snk_plot, 'b-', lw=2, label='Standard Normal Distribution (z)')

        if z_crit_upper_snk_approx is not None and not np.isnan(z_crit_upper_snk_approx):
            x_fill_upper = np.linspace(z_crit_upper_snk_approx, plot_max_z_snk, 100)
            ax_snk_approx.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, 
                                  label=f'Crit. Region (α={alpha_snk:.8f})')
            ax_snk_approx.axvline(z_crit_upper_snk_approx, color='red', linestyle='--', lw=1)
        
        ax_snk_approx.axvline(test_stat_snk_q_as_z, color='green', linestyle='-', lw=2, label=f'Input Stat (as z) = {test_stat_snk_q_as_z:.3f}')
        ax_snk_approx.set_title(f'Normal Approximation for SNK (α={alpha_snk:.8f})')
        ax_snk_approx.set_xlabel('Value (Treated as z-score)')
        ax_snk_approx.set_ylabel('Probability Density')
        ax_snk_approx.legend(); ax_snk_approx.grid(True); st.pyplot(fig_snk_approx)

        st.subheader("Standard Normal Table: Cumulative P(Z < z)")
        st.markdown("This table shows the area to the left of a given z-score. The one-tailed z-critical value (derived from your alpha) is used for highlighting.")

        z_crit_for_table_snk = z_crit_upper_snk_approx if z_crit_upper_snk_approx is not None else 0.0
        if z_crit_for_table_snk is None or np.isnan(z_crit_for_table_snk): z_crit_for_table_snk = 0.0

        all_z_row_labels_snk = [f"{val:.1f}" for val in np.round(np.arange(-3.4, 3.5, 0.1), 1)]
        z_col_labels_str_snk = [f"{val:.2f}" for val in np.round(np.arange(0.00, 0.10, 0.01), 2)]
        
        z_target_for_table_row_numeric_snk = round(z_crit_for_table_snk, 1) 

        try:
            closest_row_idx_snk = min(range(len(all_z_row_labels_snk)), key=lambda i: abs(float(all_z_row_labels_snk[i]) - z_target_for_table_row_numeric_snk))
        except ValueError: 
            closest_row_idx_snk = len(all_z_row_labels_snk) // 2

        window_size_z_snk = 5
        start_idx_z_snk = max(0, closest_row_idx_snk - window_size_z_snk)
        end_idx_z_snk = min(len(all_z_row_labels_snk), closest_row_idx_snk + window_size_z_snk + 1)
        z_table_display_rows_str_snk = all_z_row_labels_snk[start_idx_z_snk:end_idx_z_snk]

        table_data_z_lookup_snk = []
        for z_r_str_idx in z_table_display_rows_str_snk:
            z_r_val = float(z_r_str_idx)
            row = { 'z': z_r_str_idx } 
            for z_c_str_idx in z_col_labels_str_snk:
                z_c_val = float(z_c_str_idx)
                current_z_val = round(z_r_val + z_c_val, 2)
                prob = stats.norm.cdf(current_z_val)
                row[z_c_str_idx] = format_value_for_display(prob, decimals=4)
            table_data_z_lookup_snk.append(row)
        
        df_z_lookup_table_snk = pd.DataFrame(table_data_z_lookup_snk).set_index('z')
        
        # Use the generic z-table styling function
        st.markdown(df_z_lookup_table_snk.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                               {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(lambda df: style_z_lookup_table_generic(df, z_crit_for_table_snk), axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows P(Z < z). Highlighted cell, row, and column are closest to the z-critical value derived from α={alpha_snk:.8f}.")

    with col2: 
        st.subheader("P-value Calculation Explanation (Normal Approximation)")
        p_val_calc_snk_approx = float('nan')
        
        # SNK is essentially a series of one-tailed (upper) comparisons for q
        # So, for z-approximation, we use one-tailed sf
        p_val_calc_snk_approx = stats.norm.sf(test_stat_snk_q_as_z)
        p_val_calc_snk_approx = min(p_val_calc_snk_approx, 1.0) if not np.isnan(p_val_calc_snk_approx) else float('nan')

        st.markdown(f"""
        The input statistic ({test_stat_snk_q_as_z:.3f}) is treated as a z-score.
        The p-value is derived from the standard normal distribution (one-tailed, upper):
        * `P(Z ≥ {test_stat_snk_q_as_z:.3f})`
        **Disclaimer**: This is a rough approximation and not a standard Newman-Keuls p-value or decision process.
        """)

        st.subheader("Summary (Normal Approximation)")
        
        crit_val_snk_display = format_value_for_display(z_crit_upper_snk_approx)
        decision_crit_snk_approx = False
        comparison_crit_str_snk = "N/A"

        if z_crit_upper_snk_approx is not None and not np.isnan(z_crit_upper_snk_approx):
            decision_crit_snk_approx = test_stat_snk_q_as_z > z_crit_upper_snk_approx
        comparison_crit_str_snk = f"Input Stat (as z) ({test_stat_snk_q_as_z:.3f}) {' > ' if decision_crit_snk_approx else ' ≤ '} z_crit ({format_value_for_display(z_crit_upper_snk_approx)})"
            
        decision_p_alpha_snk_approx = p_val_calc_snk_approx < alpha_snk if not np.isnan(p_val_calc_snk_approx) else False
            
        st.markdown(f"""
        1.  **Approximate Critical z-value (One-tailed right)**: {crit_val_snk_display}
            * *Significance level (α)*: {alpha_snk:.8f}
        2.  **Input Statistic (treated as z-score)**: {test_stat_snk_q_as_z:.3f}
            * *Approximate p-value (from z-dist)*: {format_value_for_display(p_val_calc_snk_approx, decimals=4)} ({apa_p_value(p_val_calc_snk_approx)})
        3.  **Decision (Approx. Critical Value Method)**: H₀ (no difference for this specific comparison) is **{'rejected' if decision_crit_snk_approx else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_snk}.
        4.  **Decision (Approx. p-value Method)**: H₀ (no difference for this specific comparison) is **{'rejected' if decision_p_alpha_snk_approx else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_snk_approx)} is {'less than' if decision_p_alpha_snk_approx else 'not less than'} α ({alpha_snk:.8f}).
        5.  **APA 7 Style Report (using Normal Approximation for one comparison)**:
            Using a normal approximation for a Newman-Keuls-like comparison spanning p={p_means_spanned} means, an input statistic of {test_stat_snk_q_as_z:.2f} (total k={k_snk_total_groups}, df<sub>error</sub>={df_error_snk}) yielded an approximate {apa_p_value(p_val_calc_snk_approx)}. The null hypothesis of no difference for this specific comparison was {'rejected' if decision_p_alpha_snk_approx else 'not rejected'} at α = {alpha_snk:.2f}. (Note: This is a z-distribution based approximation, not a standard Newman-Keuls test).
        """)
# --- Tab 18: Jonckheere-Terpstra Test Table ---
def tab_jonckheere_terpstra():
    st.header("Jonckheere-Terpstra Test (Normal Approximation)")
    st.markdown("""
    Tests for an ordered difference among medians of independent groups (e.g., Group1 < Group2 < Group3).
    For larger sample sizes, the J statistic is approximately normally distributed.
    `scipy.stats.jonckheere_terpstra` provides the J statistic and a p-value.
    This tab focuses on interpreting a calculated J via normal approximation.
    """)
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_jt = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_jt_input")
        # For simplicity, we'll ask for the calculated J, mean_J, and sd_J directly
        # as calculating them from raw data or group sizes is complex for a simple table app.
        test_stat_j = st.number_input("Your Calculated J-statistic", value=100.0, format="%.1f", key="test_stat_j_input")
        mean_j = st.number_input("Expected Mean of J (μ_J) under H₀", value=90.0, format="%.1f", help="Calculated as N(N-1)/4 where N is total sample size, if no ties.")
        sd_j = st.number_input("Standard Deviation of J (σ_J) under H₀", value=10.0, format="%.1f", min_value=0.001, help="Complex formula, depends on N and group sizes n_i. See statistical texts.")
        
        tail_jt = st.radio("Tail Selection (Alternative Hypothesis)", 
                          ("One-tailed (ordered, e.g., μ₁≤μ₂≤...≤μₖ with at least one <)", 
                           "Two-tailed (any ordered difference or its reverse)"), 
                          key="tail_jt_radio", index=0)

        z_calc_jt = float('nan')
        if sd_j > 0:
            # Continuity correction can be +/- 1 for J
            # For simplicity, not applying it here, but noting it.
            z_calc_jt = (test_stat_j - mean_j) / sd_j
        
        st.markdown(f"**Approximate z-statistic (from J):** {format_value_for_display(z_calc_jt)}")
        st.caption("Note: A continuity correction of +/- 1 for J is sometimes applied before calculating z. `scipy.stats.jonckheere_terpstra` handles this and ties.")

        st.subheader("Normal Distribution Plot (Approximation for J)")
        fig_jt, ax_jt = plt.subplots(figsize=(8,5))
        
        crit_z_upper_jt_plot, crit_z_lower_jt_plot = None, None
        if tail_jt == "Two-tailed (any ordered difference or its reverse)": 
            crit_z_upper_jt_plot = stats.norm.ppf(1 - alpha_jt / 2)
            crit_z_lower_jt_plot = stats.norm.ppf(alpha_jt / 2)
        else: # One-tailed (ordered) - typically means J is expected to be large
            crit_z_upper_jt_plot = stats.norm.ppf(1 - alpha_jt)
            # If a specific reverse order was hypothesized, one might use lower tail.
            # But typically, "ordered" implies an increasing or decreasing trend tested with upper tail of J.

        plot_min_z_jt = min(stats.norm.ppf(0.00000001), z_calc_jt - 2 if not np.isnan(z_calc_jt) else -2, -4.0)
        plot_max_z_jt = max(stats.norm.ppf(0.99999999), z_calc_jt + 2 if not np.isnan(z_calc_jt) else 2, 4.0)
        if not np.isnan(z_calc_jt) and abs(z_calc_jt) > 3.5:
            plot_min_z_jt = min(plot_min_z_jt, z_calc_jt - 0.5)
            plot_max_z_jt = max(plot_max_z_jt, z_calc_jt + 0.5)
        
        x_norm_jt_plot = np.linspace(plot_min_z_jt, plot_max_z_jt, 500)
        y_norm_jt_plot = stats.norm.pdf(x_norm_jt_plot)
        ax_jt.plot(x_norm_jt_plot, y_norm_jt_plot, 'b-', lw=2, label='Standard Normal Distribution (z)')

        if crit_z_upper_jt_plot is not None and not np.isnan(crit_z_upper_jt_plot):
            x_fill_upper = np.linspace(crit_z_upper_jt_plot, plot_max_z_jt, 100)
            ax_jt.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, label=f'Crit. Region (α={alpha_jt/(2 if tail_jt.startswith("Two-tailed") else 1):.8f})')
            ax_jt.axvline(crit_z_upper_jt_plot, color='red', linestyle='--', lw=1)
        if crit_z_lower_jt_plot is not None and not np.isnan(crit_z_lower_jt_plot) and tail_jt.startswith("Two-tailed"):
            x_fill_lower = np.linspace(plot_min_z_jt, crit_z_lower_jt_plot, 100)
            ax_jt.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.5)
            ax_jt.axvline(crit_z_lower_jt_plot, color='red', linestyle='--', lw=1)
        
        if not np.isnan(z_calc_jt):
            ax_jt.axvline(z_calc_jt, color='green', linestyle='-', lw=2, label=f'Approx. z_calc = {z_calc_jt:.3f}')
        ax_jt.set_title('Normal Approximation for Jonckheere-Terpstra J'); ax_jt.legend(); ax_jt.grid(True); st.pyplot(fig_jt)

        st.subheader("Standard Normal Table: Cumulative P(Z < z)")
        st.markdown("This table shows the area to the left of a given z-score. The z-critical value (derived from your alpha and tail selection) is used for highlighting.")
        
        z_crit_for_table_jt = crit_z_upper_jt_plot if tail_jt.startswith("One-tailed") else \
                              (crit_z_upper_jt_plot if crit_z_upper_jt_plot is not None and tail_jt.startswith("Two-tailed") else 0.0)
        if z_crit_for_table_jt is None or np.isnan(z_crit_for_table_jt): z_crit_for_table_jt = 0.0

        all_z_row_labels_jt = [f"{val:.1f}" for val in np.round(np.arange(-3.4, 3.5, 0.1), 1)]
        z_col_labels_str_jt = [f"{val:.2f}" for val in np.round(np.arange(0.00, 0.10, 0.01), 2)]
        
        z_target_for_table_row_numeric_jt = round(z_crit_for_table_jt, 1) 

        try:
            closest_row_idx_jt = min(range(len(all_z_row_labels_jt)), key=lambda i: abs(float(all_z_row_labels_jt[i]) - z_target_for_table_row_numeric_jt))
        except ValueError: 
            closest_row_idx_jt = len(all_z_row_labels_jt) // 2

        window_size_z_jt = 5
        start_idx_z_jt = max(0, closest_row_idx_jt - window_size_z_jt)
        end_idx_z_jt = min(len(all_z_row_labels_jt), closest_row_idx_jt + window_size_z_jt + 1)
        z_table_display_rows_str_jt = all_z_row_labels_jt[start_idx_z_jt:end_idx_z_jt]

        table_data_z_lookup_jt = []
        for z_r_str_idx in z_table_display_rows_str_jt:
            z_r_val = float(z_r_str_idx)
            row = { 'z': z_r_str_idx } 
            for z_c_str_idx in z_col_labels_str_jt:
                z_c_val = float(z_c_str_idx)
                current_z_val = round(z_r_val + z_c_val, 2)
                prob = stats.norm.cdf(current_z_val)
                row[z_c_str_idx] = format_value_for_display(prob, decimals=4)
            table_data_z_lookup_jt.append(row)
        
        df_z_lookup_table_jt = pd.DataFrame(table_data_z_lookup_jt).set_index('z')
        
        st.markdown(df_z_lookup_table_jt.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                               {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(lambda df: style_z_lookup_table_generic(df, z_crit_for_table_jt), axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows P(Z < z). Highlighted cell, row, and column are closest to the z-critical value derived from α={alpha_jt:.8f} and tail selection.")

    with col2: 
        st.subheader("P-value Calculation Explanation (Normal Approximation)")
        p_val_calc_jt = float('nan')
        
        if not np.isnan(z_calc_jt):
            if tail_jt.startswith("Two-tailed"): 
                p_val_calc_jt = 2 * stats.norm.sf(abs(z_calc_jt))
            else: # One-tailed (ordered) - typically upper tail for J
                p_val_calc_jt = stats.norm.sf(z_calc_jt)
            p_val_calc_jt = min(p_val_calc_jt, 1.0) if not np.isnan(p_val_calc_jt) else float('nan')
        
        st.markdown(f"""
        The J statistic ({test_stat_j:.1f}) is converted to an approximate z-statistic ({format_value_for_display(z_calc_jt)}) using its expected mean and standard deviation under H₀.
        The p-value is then derived from the standard normal distribution.
        * **Two-tailed**: `2 * P(Z ≥ |{format_value_for_display(z_calc_jt)}|)`
        * **One-tailed (ordered)**: `P(Z ≥ {format_value_for_display(z_calc_jt)})` (assuming expected order implies larger J)
        """)

        st.subheader("Summary (Normal Approximation)")
        crit_val_z_display_jt = "N/A"
        if tail_jt.startswith("Two-tailed"): crit_val_z_display_jt = f"±{format_value_for_display(crit_z_upper_jt_plot)}"
        else: crit_val_z_display_jt = format_value_for_display(crit_z_upper_jt_plot)
        
        decision_crit_jt = False
        comparison_crit_str_jt = "N/A"

        if not np.isnan(z_calc_jt):
            if tail_jt.startswith("Two-tailed") and crit_z_upper_jt_plot is not None:
                decision_crit_jt = abs(z_calc_jt) > crit_z_upper_jt_plot
                comparison_crit_str_jt = f"|Approx. z_calc ({abs(z_calc_jt):.3f})| {' > ' if decision_crit_jt else ' ≤ '} z_crit ({format_value_for_display(crit_z_upper_jt_plot)})"
            elif tail_jt.startswith("One-tailed") and crit_z_upper_jt_plot is not None: # Assuming upper tail for ordered
                decision_crit_jt = z_calc_jt > crit_z_upper_jt_plot
                comparison_crit_str_jt = f"Approx. z_calc ({z_calc_jt:.3f}) {' > ' if decision_crit_jt else ' ≤ '} z_crit ({format_value_for_display(crit_z_upper_jt_plot)})"
        
        decision_p_alpha_jt = p_val_calc_jt < alpha_jt if not np.isnan(p_val_calc_jt) else False
        
        st.markdown(f"""
        1.  **Critical z-value ({tail_jt})**: {crit_val_z_display_jt}
            * *Significance level (α)*: {alpha_jt:.8f}
        2.  **Calculated J-statistic**: {test_stat_j:.1f} (Approx. z-statistic: {format_value_for_display(z_calc_jt)})
            * *Calculated p-value (Normal Approx.)*: {format_value_for_display(p_val_calc_jt, decimals=4)} ({apa_p_value(p_val_calc_jt)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_jt else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_jt}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_jt else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_jt)} is {'less than' if decision_p_alpha_jt else 'not less than'} α ({alpha_jt:.8f}).
        5.  **APA 7 Style Report (based on Normal Approximation)**:
            A Jonckheere-Terpstra test indicated {'' if decision_p_alpha_jt else 'no '}statistically significant ordered trend across the group medians, *J* = {test_stat_j:.1f}, *z* = {format_value_for_display(z_calc_jt, decimals=2)}, {apa_p_value(p_val_calc_jt)}. The null hypothesis was {'rejected' if decision_p_alpha_jt else 'not rejected'} at α = {alpha_jt:.2f}.
        """)

# --- Tab 19: Cochran’s Q Test Table ---
def tab_cochrans_q():
    st.header("Cochran’s Q Test (Chi-square Approximation)")
    st.markdown("""
    Tests for differences between three or more matched sets of frequencies or proportions for dichotomous (binary) outcomes.
    The Q statistic is approximately distributed as a Chi-square (χ²) distribution with `df = k - 1`.
    """)
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_cq = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.5, value=0.05, step=0.00000001, format="%.8f", key="alpha_cq_input")
        k_conditions_cq = st.number_input("Number of Conditions/Treatments (k)", min_value=2, value=3, step=1, key="k_conditions_cq_input") # df = k-1
        
        df_cq = k_conditions_cq - 1
        st.markdown(f"Degrees of Freedom (df) = k - 1 = {df_cq}")
        test_stat_q_cochran = st.number_input("Calculated Cochran's Q-statistic", value=float(df_cq if df_cq > 0 else 0.5), format="%.3f", min_value=0.0, key="test_stat_q_cochran_input")
        st.caption("Note: Chi-square approximation is generally better for larger numbers of subjects/blocks.")


        st.subheader("Chi-square Distribution Plot (Approximation for Q)")
        fig_cq, ax_cq = plt.subplots(figsize=(8,5))
        crit_val_chi2_cq_plot = None 
        
        if df_cq > 0:
            crit_val_chi2_cq_plot = stats.chi2.ppf(1 - alpha_cq, df_cq)
            plot_min_chi2_cq = 0.001
            plot_max_chi2_cq = max(stats.chi2.ppf(0.999, df_cq) if df_cq > 0 else 10.0, test_stat_q_cochran * 1.5, 10.0)
            if test_stat_q_cochran > (stats.chi2.ppf(0.999, df_cq) if df_cq > 0 else 10.0) * 1.2:
                plot_max_chi2_cq = test_stat_q_cochran * 1.2

            x_chi2_cq_plot = np.linspace(plot_min_chi2_cq, plot_max_chi2_cq, 500)
            y_chi2_cq_plot = stats.chi2.pdf(x_chi2_cq_plot, df_cq)
            ax_cq.plot(x_chi2_cq_plot, y_chi2_cq_plot, 'b-', lw=2, label=f'χ²-distribution (df={df_cq})')

            if isinstance(crit_val_chi2_cq_plot, (int, float)) and not np.isnan(crit_val_chi2_cq_plot):
                x_fill_upper_cq = np.linspace(crit_val_chi2_cq_plot, plot_max_chi2_cq, 100)
                ax_cq.fill_between(x_fill_upper_cq, stats.chi2.pdf(x_fill_upper_cq, df_cq), color='red', alpha=0.5, label=f'α = {alpha_cq:.8f}')
                ax_cq.axvline(crit_val_chi2_cq_plot, color='red', linestyle='--', lw=1, label=f'χ²_crit = {crit_val_chi2_cq_plot:.3f}')
            
            ax_cq.axvline(test_stat_q_cochran, color='green', linestyle='-', lw=2, label=f'Q_calc = {test_stat_q_cochran:.3f}')
            ax_cq.set_title(f'χ²-Approximation for Cochran\'s Q (df={df_cq})')
        else:
            ax_cq.text(0.5, 0.5, "df must be > 0 (k > 1 for meaningful test)", ha='center', va='center')
            ax_cq.set_title('Plot Unavailable (df=0)')
            
        ax_cq.legend(); ax_cq.grid(True); st.pyplot(fig_cq)
        
        st.subheader("Critical χ²-Values (Upper Tail)")
        all_df_chi2_options_cq = list(range(1, 31)) + [35, 40, 45, 50, 60, 70, 80, 90, 100]
        table_df_window_chi2_cq = get_dynamic_df_window(all_df_chi2_options_cq, df_cq, window_size=5)
        table_alpha_cols_chi2_cq = [0.10, 0.05, 0.025, 0.01, 0.005]

        chi2_table_rows_cq = []
        for df_iter_val in table_df_window_chi2_cq: 
            df_iter_calc = int(df_iter_val)
            row_data = {'df': str(df_iter_val)}
            for alpha_c in table_alpha_cols_chi2_cq:
                cv = stats.chi2.ppf(1 - alpha_c, df_iter_calc) if df_iter_calc > 0 else float('nan')
                row_data[f"α = {alpha_c:.3f}"] = format_value_for_display(cv)
            chi2_table_rows_cq.append(row_data)
        df_chi2_table_cq = pd.DataFrame(chi2_table_rows_cq).set_index('df')

        # Use the generic Chi-square table styling (assuming it's defined in main script or adapt)
        def style_chi2_table_cq(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_str = str(df_cq) 

            if selected_df_str in df_to_style.index:
                style.loc[selected_df_str, :] = 'background-color: lightblue;'
            
            closest_alpha_col_val = min(table_alpha_cols_chi2_cq, key=lambda x: abs(x - alpha_cq))
            highlight_col_name = f"α = {closest_alpha_col_val:.3f}"

            if highlight_col_name in df_to_style.columns:
                for r_idx in df_to_style.index:
                    current_r_style = style.loc[r_idx, highlight_col_name]
                    style.loc[r_idx, highlight_col_name] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                if selected_df_str in df_to_style.index:
                    current_c_style = style.loc[selected_df_str, highlight_col_name]
                    style.loc[selected_df_str, highlight_col_name] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style

        if df_cq > 0:
            st.markdown(df_chi2_table_cq.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                                {'selector': 'td', 'props': [('text-align', 'center')]}])
                                         .apply(style_chi2_table_cq, axis=None).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows upper-tail critical χ²-values. Highlighted for df={df_cq} and α closest to your test.")
        else:
            st.warning("df must be > 0 to generate table (k > 1).")

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value is P(χ² ≥ Q_calc) assuming H₀ (proportions are equal across conditions) is true.
        * `P(χ² ≥ {test_stat_q_cochran:.3f}) = stats.chi2.sf({test_stat_q_cochran:.3f}, df={df_cq})` (if df > 0)
        """)

        st.subheader("Summary")
        p_val_calc_cq_num = float('nan') 
        decision_crit_cq = False
        comparison_crit_str_cq = "Test not valid (df must be > 0)"
        decision_p_alpha_cq = False
        apa_Q_cochran_stat = f"Q({df_cq if df_cq > 0 else 'N/A'}) = {format_value_for_display(test_stat_q_cochran, decimals=2)}"
        
        summary_crit_val_chi2_cq_display_str = "N/A (df=0)"
        if df_cq > 0:
            p_val_calc_cq_num = stats.chi2.sf(test_stat_q_cochran, df_cq) 
            if isinstance(crit_val_chi2_cq_plot, (int,float)) and not np.isnan(crit_val_chi2_cq_plot):
                summary_crit_val_chi2_cq_display_str = f"{crit_val_chi2_cq_plot:.3f}"
                decision_crit_cq = test_stat_q_cochran > crit_val_chi2_cq_plot
                comparison_crit_str_cq = f"Q({test_stat_q_cochran:.3f}) > χ²_crit({crit_val_chi2_cq_plot:.3f})" if decision_crit_cq else f"Q({test_stat_q_cochran:.3f}) ≤ χ²_crit({crit_val_chi2_cq_plot:.3f})"
            else: 
                summary_crit_val_chi2_cq_display_str = "N/A (calc error)"
                comparison_crit_str_cq = "Comparison not possible (critical value is N/A or NaN)"

            if isinstance(p_val_calc_cq_num, (int, float)) and not np.isnan(p_val_calc_cq_num):
                decision_p_alpha_cq = p_val_calc_cq_num < alpha_cq
        
        apa_p_val_calc_cq_str = apa_p_value(p_val_calc_cq_num)

        st.markdown(f"""
        1.  **Critical χ²-value (df={df_cq})**: {summary_crit_val_chi2_cq_display_str}
            * *Associated p-value (α)*: {alpha_cq:.8f}
        2.  **Calculated Cochran's Q-statistic**: {format_value_for_display(test_stat_q_cochran)}
            * *Calculated p-value (from χ² approx.)*: {format_value_for_display(p_val_calc_cq_num, decimals=4)} ({apa_p_val_calc_cq_str})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_cq else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_cq}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_cq else 'not rejected'}**.
            * *Reason*: {apa_p_val_calc_cq_str} is {'less than' if decision_p_alpha_cq else 'not less than'} α ({alpha_cq:.8f}).
        5.  **APA 7 Style Report**:
            A Cochran’s Q test indicated {'' if decision_p_alpha_cq else 'no '}statistically significant difference in proportions across the k={k_conditions_cq} conditions, {apa_Q_cochran_stat}, {apa_p_val_calc_cq_str}. The null hypothesis was {'rejected' if decision_p_alpha_cq else 'not rejected'} at α = {alpha_cq:.2f}.
        """, unsafe_allow_html=True)

# --- Tab 20: Kolmogorov-Smirnov Test Table ---
def tab_kolmogorov_smirnov():
    st.header("Kolmogorov-Smirnov (K-S) Test Critical D Values")
    st.markdown("""
    The Kolmogorov-Smirnov test assesses the goodness of fit between an empirical distribution function (EDF) 
    of a sample and a theoretical cumulative distribution function (CDF), or between the EDFs of two samples.
    The test statistic is D, the maximum absolute difference between the CDFs.
    Critical D values depend on sample size(s) and alpha. 
    `scipy.stats.kstest` (one-sample) and `scipy.stats.ks_2samp` (two-sample) calculate D and its p-value.
    This tab provides approximate critical D values for common alpha levels.
    """)
    
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_ks = st.number_input("Alpha (α)", min_value=0.00000001, max_value=0.20, value=0.05, step=0.00000001, format="%.8f", key="alpha_ks_input")
        ks_test_type = st.radio("K-S Test Type", ("One-sample", "Two-sample (equal n, approx.)"), key="ks_test_type_radio")

        if ks_test_type == "One-sample":
            n_ks = st.number_input("Sample Size (n)", min_value=5, value=30, step=1, key="n_ks_one_sample")
            n1_ks, n2_ks = n_ks, 0 # For unified calculation later
        else: # Two-sample
            st.markdown("For two-sample K-S, critical value approximations are simpler for equal n. This tab demonstrates for equal n.")
            n_ks_two = st.number_input("Sample Size per Group (n1 = n2 = n)", min_value=5, value=30, step=1, key="n_ks_two_sample")
            n1_ks, n2_ks = n_ks_two, n_ks_two
            n_ks = n_ks_two # For table display consistency

        test_stat_d_ks = st.number_input("Your Calculated D-statistic", value=0.15, format="%.4f", min_value=0.0, max_value=1.0, key="test_stat_d_ks_input")

        st.warning("Critical D values in the table are approximations, especially for the two-sample test. For precise p-values, use `scipy.stats.kstest` or `scipy.stats.ks_2samp` with your data.")

        # Approximate K_alpha constants for one-sample K-S (D_crit = K_alpha / sqrt(n))
        K_ALPHA_ONE_SAMPLE = { 0.20: 1.07, 0.10: 1.22, 0.05: 1.36, 0.02: 1.52, 0.01: 1.63 }
        # Approximate K_alpha constants for two-sample K-S (D_crit = K_alpha * sqrt((n1+n2)/(n1*n2)) )
        # For equal n (n1=n2=n_ks), this simplifies to K_alpha * sqrt(2/n_ks)
        K_ALPHA_TWO_SAMPLE = { 0.20: 1.07, 0.10: 1.22, 0.05: 1.36, 0.02: 1.52, 0.01: 1.63 } # Same constants, different formula

        st.subheader(f"Approximate Critical D Values for K-S Test (α={alpha_ks:.8f})")
        
        # Table generation
        all_n_options_ks = list(range(5, 51, 5)) + list(range(60, 101, 10)) + [150, 200]
        table_n_window_ks = get_dynamic_df_window(all_n_options_ks, n_ks, window_size=3)
        table_alpha_ks_cols = [0.20, 0.10, 0.05, 0.02, 0.01]

        ks_table_rows = []
        for n_iter in table_n_window_ks:
            n_iter_calc = int(n_iter)
            row_data = {'n (or n per group)': str(n_iter_calc)}
            for alpha_col in table_alpha_ks_cols:
                d_crit_cell = float('nan')
                if ks_test_type == "One-sample":
                    if n_iter_calc > 0 and alpha_col in K_ALPHA_ONE_SAMPLE:
                        d_crit_cell = K_ALPHA_ONE_SAMPLE[alpha_col] / math.sqrt(n_iter_calc)
                else: # Two-sample (equal n approx)
                    if n_iter_calc > 0 and alpha_col in K_ALPHA_TWO_SAMPLE:
                         d_crit_cell = K_ALPHA_TWO_SAMPLE[alpha_col] * math.sqrt(2 / n_iter_calc)
                row_data[f"α = {alpha_col:.2f}"] = format_value_for_display(d_crit_cell, decimals=4)
            ks_table_rows.append(row_data)
        
        df_ks_table = pd.DataFrame(ks_table_rows).set_index('n (or n per group)')

        def style_ks_table(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_n_str = str(n_ks)
            
            # Find closest n in table for highlighting row
            available_n_in_table = [int(idx_str) for idx_str in df_to_style.index if idx_str.isdigit()]
            closest_n_highlight = min(available_n_in_table, key=lambda x:abs(x-n_ks)) if available_n_in_table else -1
            selected_n_highlight_str = str(closest_n_highlight if closest_n_highlight != -1 else n_ks)


            if selected_n_highlight_str in df_to_style.index: 
                style.loc[selected_n_highlight_str, :] = 'background-color: lightblue;'
            
            closest_alpha_col_val_ks = min(table_alpha_ks_cols, key=lambda x: abs(x - alpha_ks))
            highlight_col_name_ks = f"α = {closest_alpha_col_val_ks:.2f}" # Match table header format

            if highlight_col_name_ks in df_to_style.columns:
                for r_idx in df_to_style.index:
                     current_r_style = style.loc[r_idx, highlight_col_name_ks]
                     style.loc[r_idx, highlight_col_name_ks] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                if selected_n_highlight_str in df_to_style.index: 
                    current_c_style = style.loc[selected_n_highlight_str, highlight_col_name_ks]
                    style.loc[selected_n_highlight_str, highlight_col_name_ks] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style

        if not df_ks_table.empty:
            st.markdown(df_ks_table.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                          {'selector': 'td', 'props': [('text-align', 'center')]}])
                                         .apply(style_ks_table, axis=None).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows approximate critical D values. Highlighted for n closest to {n_ks} and α closest to your input.")
        else:
            st.warning("K-S critical value table could not be generated.")

    with col2:
        st.subheader("Summary")
        critical_d_ks = float('nan')
        if ks_test_type == "One-sample":
            if n_ks > 0 and alpha_ks in K_ALPHA_ONE_SAMPLE:
                critical_d_ks = K_ALPHA_ONE_SAMPLE[alpha_ks] / math.sqrt(n_ks)
            source_note_ks = f"One-sample K-S approx. (K_α={K_ALPHA_ONE_SAMPLE.get(alpha_ks, 'N/A')})"
        else: # Two-sample
            if n1_ks > 0 and alpha_ks in K_ALPHA_TWO_SAMPLE: # n1_ks is n_per_group here
                 critical_d_ks = K_ALPHA_TWO_SAMPLE[alpha_ks] * math.sqrt(2 / n1_ks)
            source_note_ks = f"Two-sample K-S approx. for equal n (K_α={K_ALPHA_TWO_SAMPLE.get(alpha_ks, 'N/A')})"
        
        st.markdown(f"**Approximate Critical D (α={alpha_ks:.8f}, n={n_ks if n2_ks == 0 else f'{n1_ks},{n2_ks}'})**: {format_value_for_display(critical_d_ks, decimals=4)}")
        st.caption(source_note_ks)

        decision_ks = "N/A"
        comparison_str_ks = "Compare your observed D to the critical D."
        if not np.isnan(test_stat_d_ks) and not np.isnan(critical_d_ks):
            if test_stat_d_ks > critical_d_ks:
                decision_ks = "Reject H₀ (distributions differ significantly)"
            else:
                decision_ks = "Fail to reject H₀ (no significant difference between distributions)"
            comparison_str_ks = f"Observed D ({test_stat_d_ks:.4f}) {' > ' if test_stat_d_ks > critical_d_ks else ' ≤ '} Critical D ({format_value_for_display(critical_d_ks, decimals=4)})"
        
        st.markdown(f"""
        1.  **Your Calculated D-statistic**: {test_stat_d_ks:.4f}
        2.  **Approximate Critical D-value**: {format_value_for_display(critical_d_ks, decimals=4)}
        3.  **Decision**: {decision_ks}
            * *Reason*: {comparison_str_ks}.
        """)
        st.markdown("""
        **Interpretation**: If your calculated D > Critical D, reject the null hypothesis that the distributions are the same (or that the sample follows the specified theoretical distribution).
        """)

# --- Tab 21: Lilliefors Test Table ---
def tab_lilliefors():
    st.header("Lilliefors Test for Normality (Illustrative Critical Values)")
    st.markdown("""
    The Lilliefors test is a modification of the Kolmogorov-Smirnov test used specifically for testing if a sample 
    comes from a normally distributed population when the mean and standard deviation of the population are **unknown** and are estimated from the sample. Critical values are different from the standard K-S test.
    **The table provided here is a very small, illustrative placeholder for demonstration purposes only.**
    For accurate research, consult comprehensive Lilliefors test tables or use statistical software 
    (e.g., `statsmodels.stats.diagnostic.lilliefors` in Python).
    """)
    col1, col2 = st.columns([2, 1.5])

    # Highly abridged placeholder table for Lilliefors critical values
    # Source: Adapted from various statistical tables (e.g., Conover, Practical Nonparametric Statistics)
    # Key is sample size (n)
    LILLIEFORS_CRITICAL_VALUES = {
        # alpha: 0.20,  0.15,  0.10,  0.05,  0.01
        5:      [0.300, 0.315, 0.337, 0.361, 0.405],
        10:     [0.220, 0.230, 0.242, 0.262, 0.294],
        15:     [0.180, 0.190, 0.201, 0.219, 0.247],
        20:     [0.161, 0.167, 0.174, 0.190, 0.213], # Corrected value for alpha=0.05, n=20 is often cited around 0.190
        25:     [0.142, 0.149, 0.158, 0.173, 0.195],
        30:     [0.131, 0.136, 0.144, 0.161, 0.187],
        # For n > 30, approximations like K-S with adjustments or specific formulas are used.
        # e.g., for alpha=0.05, approx D_crit ~ 0.886/sqrt(n) for large n
    }
    LILLIEFORS_ALPHAS = [0.20, 0.15, 0.10, 0.05, 0.01]


    def get_lilliefors_critical_value(n_sample, alpha_level, table, alphas_in_table):
        if n_sample not in table:
            # Very simple approximation for n > 30 for demonstration
            if n_sample > 30:
                if alpha_level == 0.05: val = 0.886 / math.sqrt(n_sample)
                elif alpha_level == 0.01: val = 1.031 / math.sqrt(n_sample)
                elif alpha_level == 0.10: val = 0.805 / math.sqrt(n_sample)
                else: return float('nan'), "Alpha not in simple approx. for n>30"
                return val, f"Approximation for n > 30 (α={alpha_level})"
            return float('nan'), f"Sample size n={n_sample} not in illustrative table."
        
        try:
            alpha_idx = alphas_in_table.index(alpha_level)
            crit_val = table[n_sample][alpha_idx]
            return crit_val, f"Illustrative table (n={n_sample}, α={alpha_level})"
        except (ValueError, IndexError):
            return float('nan'), f"Alpha α={alpha_level} not in illustrative table columns."


    with col1:
        st.subheader("Inputs")
        alpha_lillie = st.selectbox("Alpha (α)", LILLIEFORS_ALPHAS, index=LILLIEFORS_ALPHAS.index(0.05), key="alpha_lillie_input")
        n_lillie = st.number_input("Sample Size (n)", min_value=4, max_value=30, value=20, step=1, key="n_lillie_input", help="Illustrative table limited to n<=30. Approximations used for n>30.")
        test_stat_d_lillie = st.number_input("Your Calculated Lilliefors D-statistic", value=0.10, format="%.4f", min_value=0.0, max_value=1.0, key="test_stat_d_lillie_input")
        
        st.subheader(f"Illustrative Lilliefors Critical D Values for α={alpha_lillie}")
        
        # Display a portion of the Lilliefors table
        lillie_table_display_data = []
        # Show a few n values around the selected n
        all_n_lillie_options = sorted(LILLIEFORS_CRITICAL_VALUES.keys())
        table_n_lillie_window = get_dynamic_df_window(all_n_lillie_options, n_lillie, window_size=2)


        for n_val in table_n_lillie_window:
            if n_val in LILLIEFORS_CRITICAL_VALUES:
                row = {'n': n_val}
                for i, alpha_h in enumerate(LILLIEFORS_ALPHAS):
                    row[f'α={alpha_h:.2f}'] = LILLIEFORS_CRITICAL_VALUES[n_val][i]
                lillie_table_display_data.append(row)
        
        df_lillie_display = pd.DataFrame(lillie_table_display_data).set_index('n')

        def style_lilliefors_table(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_n_str = str(n_lillie)
            
            if selected_n_str in df_to_style.index: # Check if string version of n_lillie is in index
                style.loc[selected_n_str, :] = 'background-color: lightblue;'
            
            highlight_col_name_lillie = f"α={alpha_lillie:.2f}"
            if highlight_col_name_lillie in df_to_style.columns:
                for r_idx in df_to_style.index:
                    current_r_style = style.loc[r_idx, highlight_col_name_lillie]
                    style.loc[r_idx, highlight_col_name_lillie] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                if selected_n_str in df_to_style.index:
                    current_c_style = style.loc[selected_n_str, highlight_col_name_lillie]
                    style.loc[selected_n_str, highlight_col_name_lillie] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style

        if not df_lillie_display.empty:
            st.markdown(df_lillie_display.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                          {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_lilliefors_table, axis=None).to_html(), unsafe_allow_html=True)
        else:
            st.warning("Could not display Lilliefors illustrative table for current inputs.")


    with col2:
        st.subheader("Summary (Based on Illustrative Table/Approximation)")
        critical_d_lillie, lillie_source_note = get_lilliefors_critical_value(n_lillie, alpha_lillie, LILLIEFORS_CRITICAL_VALUES, LILLIEFORS_ALPHAS)

        st.markdown(f"**Illustrative/Approx. Critical D (α={alpha_lillie}, n={n_lillie})**: {format_value_for_display(critical_d_lillie, decimals=4)}")
        st.caption(lillie_source_note)

        decision_lillie = "N/A"
        comparison_str_lillie = "Compare your observed D to the critical D."
        if not np.isnan(test_stat_d_lillie) and not np.isnan(critical_d_lillie):
            if test_stat_d_lillie > critical_d_lillie:
                decision_lillie = "Reject H₀ (sample likely not from a normal distribution)"
            else:
                decision_lillie = "Fail to reject H₀ (no significant evidence against normality)"
            comparison_str_lillie = f"Observed D ({test_stat_d_lillie:.4f}) {' > ' if test_stat_d_lillie > critical_d_lillie else ' ≤ '} Critical D ({format_value_for_display(critical_d_lillie, decimals=4)})"
        
        st.markdown(f"""
        1.  **Your Calculated D-statistic**: {test_stat_d_lillie:.4f}
        2.  **Illustrative/Approx. Critical D-value**: {format_value_for_display(critical_d_lillie, decimals=4)}
        3.  **Decision**: {decision_lillie}
            * *Reason*: {comparison_str_lillie}.
        """)
        st.markdown("""
        **Interpretation**: If your calculated D > Critical D, reject the null hypothesis that the data come from a normally distributed population (with unspecified mean and variance).
        """)
# --- Tab 22: Grubbs’ Test Table (for identifying a single outlier) ---
def tab_grubbs_test():
    st.header("Grubbs’ Test for Outliers (Critical G Values)")
    st.markdown("""
    Grubbs' test (also known as the maximum normalized residual test) is used to detect a single outlier 
    in a univariate dataset that is assumed to come from a normally distributed population.
    The critical G value is derived from the t-distribution.
    This tab focuses on the critical value for testing if the most extreme value (max or min) is an outlier.
    """)
    
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_grubbs = st.number_input("Alpha (α) (typically two-sided for outlier test)", min_value=0.00000001, max_value=0.20, value=0.05, step=0.00000001, format="%.8f", key="alpha_grubbs_input")
        n_grubbs = st.number_input("Sample Size (N)", min_value=3, value=20, step=1, key="n_grubbs_input") # Grubbs' test requires N >= 3
        
        test_stat_g_grubbs = st.number_input("Your Calculated Grubbs' G-statistic", value=1.5, format="%.3f", min_value=0.0, key="test_stat_g_grubbs_input")
        
        st.markdown(f"**Degrees of Freedom (for underlying t-distribution)** = N - 2 = **{n_grubbs - 2 if n_grubbs >=3 else 'N/A'}**")

        # Grubbs' test critical value formula for a single outlier (two-sided alpha)
        # G_crit = ((N-1) / sqrt(N)) * sqrt(t_crit(α/(2N), N-2)² / (N - 2 + t_crit(α/(2N), N-2)²))
        
        critical_g_value = float('nan')
        t_for_g_crit = float('nan')
        df_g = n_grubbs - 2
        
        if df_g > 0 : # N must be >= 3 for df > 0
            try:
                # For a two-sided test for an outlier, alpha is divided by N (or 2N for some conventions, here using alpha/N for one specific extreme value test or alpha/2N if testing min OR max)
                # Most tables for Grubbs test provide critical values for specific alpha levels (e.g. 0.05, 0.01) directly.
                # Here, we use the t-distribution formula which often corresponds to alpha/N for a one-sided test on the most extreme point, or alpha/(2N) for a test of a specific point.
                # Let's use alpha_grubbs directly in the t-distribution as if it's the probability for the *t-value*, then transform.
                # For a single outlier (max or min), the significance level for the t-distribution is alpha/N.
                # For a two-sided Grubbs' test (is the most extreme an outlier?), use alpha. The formula uses alpha/(2N) in some derivations for the t_ppf.
                # Simplified from common critical value formula:
                # t_crit = stats.t.ppf(1 - (alpha_grubbs / (2 * n_grubbs)), df_g)
                
                # Using a common simplified critical G formula from t-distribution directly related to alpha for the G statistic
                # G_crit = ((n-1)/sqrt(n)) * t_crit(alpha/(2n), n-2)  -- this t_crit itself is complex
                # An alternative is to use pre-calculated G values or a direct formula if available for G's CDF/PPF.
                # For simplicity, we'll use the t-distribution directly for the critical t value used in the formula:
                # Critical t for Grubbs' test is often looked up for alpha/(2N) for a two-sided test
                # However, many Grubbs tables are directly for alpha (e.g. 0.05, 0.01 for the G statistic)

                # Let's use the formula based on t_ppf(1 - alpha / (2*N), N-2)
                # This tests if EITHER the min OR max is an outlier (two-sided context for the "outlierness")
                alpha_for_t = alpha_grubbs / (2 * n_grubbs)
                if df_g > 0 and alpha_for_t > 0 and alpha_for_t < 1:
                    t_for_g_crit = stats.t.ppf(1 - alpha_for_t, df_g)
                    if not np.isnan(t_for_g_crit) and (t_for_g_crit**2 + df_g) > 0:
                         critical_g_value = ((n_grubbs - 1) / math.sqrt(n_grubbs)) * \
                                           math.sqrt((t_for_g_crit**2) / (df_g + t_for_g_crit**2))
            except Exception as e:
                st.warning(f"Could not calculate critical G: {e}")

        st.markdown(f"**Calculated Critical Grubbs' G (for single outlier, α={alpha_grubbs:.8f})**: {format_value_for_display(critical_g_value, decimals=4)}")
        if not np.isnan(t_for_g_crit):
            st.caption(f"(Based on t-critical ≈ {format_value_for_display(t_for_g_crit, decimals=4)} with df={df_g} at p≈{alpha_for_t:.8f})")

        st.subheader(f"Illustrative t-Distribution Plot (df={df_g}) related to G")
        fig_g, ax_g = plt.subplots(figsize=(8,5))
        if df_g > 0:
            plot_min_g_t = -4.0
            plot_max_g_t = 4.0
            if not np.isnan(t_for_g_crit) and abs(t_for_g_crit) > 3.5:
                plot_max_g_t = abs(t_for_g_crit) + 1
                plot_min_g_t = -plot_max_g_t
            
            x_g_t_plot = np.linspace(plot_min_g_t, plot_max_g_t, 500)
            y_g_t_plot = stats.t.pdf(x_g_t_plot, df_g)
            ax_g.plot(x_g_t_plot, y_g_t_plot, 'b-', lw=2, label=f't-distribution (df={df_g})')

            if not np.isnan(t_for_g_crit):
                ax_g.axvline(t_for_g_crit, color='red', linestyle='--', lw=1, label=f't_crit (for G) ≈ {t_for_g_crit:.3f}')
                ax_g.axvline(-t_for_g_crit, color='red', linestyle='--', lw=1)
                x_fill_upper = np.linspace(t_for_g_crit, plot_max_g_t, 100)
                ax_g.fill_between(x_fill_upper, stats.t.pdf(x_fill_upper, df_g), color='red', alpha=0.3)
                x_fill_lower = np.linspace(plot_min_g_t, -t_for_g_crit, 100)
                ax_g.fill_between(x_fill_lower, stats.t.pdf(x_fill_lower, df_g), color='red', alpha=0.3)

            ax_g.set_title(f't-Distribution used in Grubbs\' Test (df={df_g})')
            ax_g.set_xlabel('t-value')
            ax_g.set_ylabel('Probability Density')
            ax_g.legend(); ax_g.grid(True)
        else:
            ax_g.text(0.5,0.5, "df must be > 0 for plot.", ha='center')
        st.pyplot(fig_g)
        st.caption("This plot shows the t-distribution. The critical G value is a transformation of a critical t-value from this distribution.")

        st.subheader("Critical Grubbs' G Values Table")
        # Table of critical G values for single outlier
        # Columns are alpha levels (typically two-sided, for "is the most extreme an outlier?")
        all_n_grubbs_options = list(range(3, 51)) + [60, 80, 100, 120, 150, 200]
        table_n_grubbs_window = get_dynamic_df_window(all_n_grubbs_options, n_grubbs, window_size=5)
        table_alpha_grubbs_cols = [0.10, 0.05, 0.025, 0.01] # Common alpha levels for G tables

        grubbs_table_rows = []
        for n_iter in table_n_grubbs_window:
            n_iter_calc = int(n_iter)
            df_iter_calc = n_iter_calc - 2
            if df_iter_calc <= 0: continue
            row_data = {'N': str(n_iter_calc)}
            for alpha_col in table_alpha_grubbs_cols:
                g_crit_cell = float('nan')
                alpha_for_t_cell = alpha_col / (2 * n_iter_calc) # For two-sided test of single outlier
                if alpha_for_t_cell > 0 and alpha_for_t_cell < 1:
                    t_crit_val_cell = stats.t.ppf(1 - alpha_for_t_cell, df_iter_calc)
                    if not np.isnan(t_crit_val_cell) and (t_crit_val_cell**2 + df_iter_calc) > 0:
                        g_crit_cell = ((n_iter_calc - 1) / math.sqrt(n_iter_calc)) * \
                                      math.sqrt((t_crit_val_cell**2) / (df_iter_calc + t_crit_val_cell**2))
                row_data[f"α = {alpha_col:.3f}"] = format_value_for_display(g_crit_cell, decimals=3)
            grubbs_table_rows.append(row_data)
        
        df_grubbs_table = pd.DataFrame(grubbs_table_rows).set_index('N')

        def style_grubbs_table(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_n_str = str(n_grubbs)
            
            if selected_n_str in df_to_style.index: 
                style.loc[selected_n_str, :] = 'background-color: lightblue;'
            
            closest_alpha_col_val = min(table_alpha_grubbs_cols, key=lambda x: abs(x - alpha_grubbs))
            highlight_col_name = f"α = {closest_alpha_col_val:.3f}"

            if highlight_col_name in df_to_style.columns:
                for r_idx in df_to_style.index:
                     current_r_style = style.loc[r_idx, highlight_col_name]
                     style.loc[r_idx, highlight_col_name] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                if selected_n_str in df_to_style.index: 
                    current_c_style = style.loc[selected_n_str, highlight_col_name]
                    style.loc[selected_n_str, highlight_col_name] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style

        if not df_grubbs_table.empty:
            st.markdown(df_grubbs_table.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                          {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_grubbs_table, axis=None).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows critical Grubbs' G values. Highlighted for N={n_grubbs} and α closest to your input.")
        else:
            st.warning("Grubbs' critical value table could not be generated.")

    with col2:
        st.subheader("Summary for Grubbs' Test (Single Outlier)")
        
        decision_grubbs = "N/A"
        comparison_str_grubbs = "Compare your observed G to the critical G."
        if not np.isnan(test_stat_g_grubbs) and not np.isnan(critical_g_value):
            if test_stat_g_grubbs > critical_g_value:
                decision_grubbs = "Reject H₀ (the most extreme value is likely an outlier)"
            else:
                decision_grubbs = "Fail to reject H₀ (no significant evidence that the most extreme value is an outlier)"
            comparison_str_grubbs = f"Observed G ({test_stat_g_grubbs:.3f}) {' > ' if test_stat_g_grubbs > critical_g_value else ' ≤ '} Critical G ({format_value_for_display(critical_g_value, decimals=3)})"
        
        # P-value for Grubbs' test is not straightforward to calculate without specialized functions or the G distribution.
        # Scipy does not have a direct Grubbs test function.
        st.markdown(f"""
        1.  **Your Calculated Grubbs' G-statistic**: {test_stat_g_grubbs:.3f}
        2.  **Critical Grubbs' G (α={alpha_grubbs:.8f}, N={n_grubbs})**: {format_value_for_display(critical_g_value, decimals=3)}
        3.  **Decision**: {decision_grubbs}
            * *Reason*: {comparison_str_grubbs}.
        """)
        st.markdown("""
        **Interpretation**: If your calculated G > Critical G, you reject the null hypothesis that there are no outliers, 
        concluding that the most extreme value is an outlier.
        Note: This tab is for detecting a *single* outlier. Different procedures/critical values are used for detecting multiple outliers. 
        The test assumes data are approximately normally distributed.
        """)

# --- Tab 23: Durbin-Watson Test Table ---
def tab_durbin_watson():
    st.header("Durbin-Watson Test (Illustrative Bounds d_L, d_U)")
    st.markdown("""
    The Durbin-Watson test is used to detect the presence of autocorrelation (a relationship between values separated 
    from each other by a given time lag) in the residuals from a regression analysis.
    The test statistic `d` ranges from 0 to 4.
    * `d ≈ 2`: No autocorrelation.
    * `d < 2`: Positive autocorrelation.
    * `d > 2`: Negative autocorrelation.
    Interpretation requires comparing `d` to lower (d_L) and upper (d_U) critical bounds.
    **The table provided here contains a very small, illustrative subset of these bounds for demonstration purposes only.**
    For accurate research, consult comprehensive Durbin-Watson tables for your specific `n`, `k'`, and alpha.
    """)
    
    col1, col2 = st.columns([2, 1.5])

    # Highly abridged placeholder table for Durbin-Watson dL and dU values (alpha = 0.05)
    # Key: (n, k') -> (dL, dU) where k' is number of predictors (excluding intercept)
    DW_TABLE_ALPHA_05 = {
        # (n, k'): [dL, dU]
        (15, 1): [1.08, 1.36], (15, 2): [0.95, 1.54], (15, 3): [0.82, 1.75], (15, 4): [0.69, 1.97], (15, 5): [0.56, 2.21],
        (20, 1): [1.20, 1.41], (20, 2): [1.10, 1.54], (20, 3): [1.00, 1.68], (20, 4): [0.90, 1.83], (20, 5): [0.79, 1.99],
        (30, 1): [1.35, 1.49], (30, 2): [1.28, 1.57], (30, 3): [1.21, 1.65], (30, 4): [1.14, 1.74], (30, 5): [1.07, 1.83],
        (50, 1): [1.50, 1.59], (50, 2): [1.46, 1.63], (50, 3): [1.42, 1.67], (50, 4): [1.38, 1.72], (50, 5): [1.34, 1.77],
        (100,1): [1.65, 1.69], (100,2): [1.63, 1.72], (100,3): [1.61, 1.74], (100,4): [1.59, 1.76], (100,5): [1.57, 1.78],
    }
    # Could add more for alpha=0.01 etc.

    def get_dw_bounds(n_obs, k_preds, alpha_level, table_05):
        if alpha_level != 0.05: # Only 0.05 table is a placeholder
            return float('nan'), float('nan'), "Only α=0.05 illustrative table available."
        
        # Find closest n in table keys
        available_n_keys = sorted(list(set(key[0] for key in table_05.keys())))
        if not available_n_keys: return float('nan'), float('nan'), "Embedded table empty."
        closest_n = min(available_n_keys, key=lambda x: abs(x - n_obs))

        # Check if k_preds is valid for the chosen n
        if (closest_n, k_preds) in table_05:
            dl, du = table_05[(closest_n, k_preds)]
            source_note = f"Used n={closest_n}, k'={k_preds} from illustrative table."
            if closest_n != n_obs: source_note += f" (Input n={n_obs})"
            return dl, du, source_note
        else:
            return float('nan'), float('nan'), f"Combination n={closest_n}, k'={k_preds} not in illustrative table."

    with col1:
        st.subheader("Inputs")
        alpha_dw = st.selectbox("Alpha (α)", [0.05, 0.01], index=0, key="alpha_dw_input", help="Illustrative table only has values for α=0.05.")
        n_dw = st.number_input("Sample Size (n, number of observations)", min_value=15, max_value=100, value=30, step=1, key="n_dw_input", help="Illustrative table limited.")
        k_prime_dw = st.number_input("Number of Predictor Variables (k', excluding intercept)", min_value=1, max_value=5, value=1, step=1, key="k_prime_dw_input", help="Illustrative table limited.")
        
        test_stat_d_dw = st.number_input("Your Calculated Durbin-Watson d-statistic", value=2.0, format="%.3f", min_value=0.0, max_value=4.0, key="test_stat_d_dw_input")
        
        st.subheader(f"Illustrative Durbin-Watson Bounds (d_L, d_U) for α={alpha_dw}")
        if alpha_dw == 0.05:
            dw_table_display_data = []
            # Show some n values around the selected n, for k' = 1 to 5
            all_n_dw_options = sorted(list(set(key[0] for key in DW_TABLE_ALPHA_05.keys())))
            table_n_dw_window = get_dynamic_df_window(all_n_dw_options, n_dw, window_size=2)

            for n_val in table_n_dw_window:
                row = {'n': n_val}
                for k_val in range(1, 6): # k' from 1 to 5
                    bounds = DW_TABLE_ALPHA_05.get((n_val, k_val))
                    if bounds:
                        row[f"k'={k_val}"] = f"{bounds[0]:.2f}, {bounds[1]:.2f}"
                    else:
                        row[f"k'={k_val}"] = "N/A"
                dw_table_display_data.append(row)
            
            df_dw_display = pd.DataFrame(dw_table_display_data).set_index('n')
            
            def style_dw_table(df_to_style):
                style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
                selected_n_str = str(n_dw)
                if selected_n_str in df_to_style.index.astype(str):
                    style.loc[int(selected_n_str), :] = 'background-color: lightblue;'
                
                selected_k_col = f"k'={k_prime_dw}"
                if selected_k_col in df_to_style.columns:
                    style.loc[:, selected_k_col] = style.loc[:, selected_k_col].astype(str) + '; background-color: lightgreen;'
                if selected_n_str in df_to_style.index.astype(str) and selected_k_col in df_to_style.columns:
                     current_style = style.loc[int(selected_n_str), selected_k_col]
                     style.loc[int(selected_n_str), selected_k_col] = (current_style + ';' if current_style and not current_style.endswith(';') else current_style) + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
                return style

            if not df_dw_display.empty:
                 st.markdown(df_dw_display.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                          {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_dw_table, axis=None).to_html(), unsafe_allow_html=True)
                 st.caption(f"Table shows (d_L, d_U) bounds. Highlighted for n closest to {n_dw} and k'={k_prime_dw}.")
            else:
                st.warning("Could not display Durbin-Watson illustrative table.")
        else:
            st.warning("Illustrative Durbin-Watson table only available for α=0.05 in this demo.")

    with col2:
        st.subheader("Interpretation Summary")
        dl, du, dw_source_note = get_dw_bounds(n_dw, k_prime_dw, alpha_dw, DW_TABLE_ALPHA_05)
        
        st.markdown(f"**Illustrative Bounds (α={alpha_dw}, n={n_dw}, k'={k_prime_dw})**: ")
        st.markdown(f"* d_L = {format_value_for_display(dl, decimals=2)}")
        st.markdown(f"* d_U = {format_value_for_display(du, decimals=2)}")
        st.caption(dw_source_note)

        interpretation_dw = "Bounds not available from illustrative table for current inputs."
        if not np.isnan(dl) and not np.isnan(du):
            if test_stat_d_dw < dl:
                interpretation_dw = "Evidence of positive first-order autocorrelation."
            elif test_stat_d_dw > du and test_stat_d_dw < (4 - du):
                interpretation_dw = "No evidence of first-order autocorrelation (test is inconclusive in the 'no decision' region if your d falls between dL and dU, or between 4-dU and 4-dL)."
            elif test_stat_d_dw > (4 - dl):
                interpretation_dw = "Evidence of negative first-order autocorrelation."
            elif test_stat_d_dw >= dl and test_stat_d_dw <= du:
                interpretation_dw = "Test is inconclusive (positive autocorrelation)."
            elif test_stat_d_dw >= (4 - du) and test_stat_d_dw <= (4 - dl):
                interpretation_dw = "Test is inconclusive (negative autocorrelation)."
            else: # Should be caught by d < (4-du)
                interpretation_dw = "No evidence of first-order autocorrelation."


        st.markdown(f"""
        **Your Calculated d-statistic**: {test_stat_d_dw:.3f}
        
        **Decision Rule (Comparing d to d_L and d_U at α={alpha_dw}):**
        * If `d < d_L`: Reject H₀ (no autocorrelation), conclude positive autocorrelation.
        * If `d > d_U` AND `d < 4 - d_U`: Fail to reject H₀ (no autocorrelation).
        * If `d_L ≤ d ≤ d_U`: Test is inconclusive regarding positive autocorrelation.
        * If `4 - d_L < d`: Reject H₀, conclude negative autocorrelation.
        * If `4 - d_U ≤ d ≤ 4 - d_L`: Test is inconclusive regarding negative autocorrelation.
        
        **Interpretation for your d = {test_stat_d_dw:.3f}**: {interpretation_dw}
        """)
        st.markdown("Note: `statsmodels.stats.stattools.durbin_watson` calculates the d-statistic from residuals.")

# --- Main app ---
def main():
    st.set_page_config(page_title="🧠 Oli's – Statistical Table Explorer - Mk. 3", layout="wide") # Updated Page Title
    st.title("🧠 Oli's – Statistical Table Explorer - Mk. 3") # Updated App Title
    st.markdown("""
    This application provides an interactive way to explore various statistical distributions and tests. 
    Select a tab to begin. On each tab, you can adjust parameters like alpha, degrees of freedom, 
    and input a calculated test statistic to see how it compares to critical values and to understand p-value calculations.
    **Note for Tukey HSD Tab**: This tab uses a simplified normal (z) distribution approximation. For accurate Tukey HSD results, ensure `statsmodels` is installed in your environment (e.g., add `statsmodels` to `requirements.txt`) and consult statistical software that properly implements the Studentized Range (q) distribution.
    Many other non-parametric and post-hoc test tabs also use approximations or illustrative tables where exact distributions/tables are complex; please see notes on individual tabs.
    """)

    tab_names = [
        "t-Distribution",               # Tab 1
        "z-Distribution",               # Tab 2
        "F-Distribution",               # Tab 3
        "Chi-square (χ²)",              # Tab 4
        "Mann-Whitney U",             # Tab 5
        "Wilcoxon Signed-Rank T",     # Tab 6
        "Binomial Test",                # Tab 7
        "Tukey HSD (z-Approx.)",      # Tab 8 - Name updated to reflect approximation
        "Kruskal-Wallis H",           # Tab 9
        "Friedman Test",                # Tab 10
        "Critical r Table",             # Tab 11 (Pearson Correlation)
        "Spearman's Rho",             # Tab 12 - New
        "Kendall's Tau",              # Tab 13 - New
        "Hartley’s F_max",            # Tab 14 - New
        "Scheffé’s Test",             # Tab 15 - New
        "Dunnett’s Test",             # Tab 16 - New
        "Newman–Keuls (z-Approx.)",   # Tab 17 - New - Name updated
        "Jonckheere–Terpstra",        # Tab 18 - New
        "Cochran’s Q Test",           # Tab 19 - New
        "Kolmogorov–Smirnov",         # Tab 20 - New
        "Lilliefors Test",              # Tab 21 - New
        "Grubbs’ Test",               # Tab 22 - New
        "Durbin–Watson Test"          # Tab 23 - New
    ]
    
    tabs = st.tabs(tab_names)

    # Existing Tabs
    with tabs[0]:
        tab_t_distribution()
    with tabs[1]:
        tab_z_distribution()
    with tabs[2]:
        tab_f_distribution()
    with tabs[3]:
        tab_chi_square_distribution()
    with tabs[4]:
        tab_mann_whitney_u()
    with tabs[5]:
        tab_wilcoxon_t()
    with tabs[6]:
        tab_binomial_test()
    with tabs[7]:
        tab_tukey_hsd() # This is the z-approximation version
    with tabs[8]:
        tab_kruskal_wallis()
    with tabs[9]:
        tab_friedman_test()
    with tabs[10]:
        tab_critical_r()
    
    # New Tabs (ensure these function names match your definitions)
    with tabs[11]:
        tab_spearmans_rho() 
    with tabs[12]:
        tab_kendalls_tau()
    with tabs[13]:
        tab_hartleys_fmax()
    with tabs[14]:
        tab_scheffe()
    with tabs[15]:
        tab_dunnetts_test()
    with tabs[16]:
        tab_newman_keuls() # This is the z-approximation version
    with tabs[17]:
        tab_jonckheere_terpstra()
    with tabs[18]:
        tab_cochrans_q()
    with tabs[19]:
        tab_kolmogorov_smirnov()
    with tabs[20]:
        tab_lilliefors()
    with tabs[21]:
        tab_grubbs_test()
    with tabs[22]:
        tab_durbin_watson()

if __name__ == "__main__":
    main()
