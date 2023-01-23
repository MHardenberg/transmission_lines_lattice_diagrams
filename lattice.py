import numpy as np
from ESHD_formulas.helper import is_even


def run_diagram(v_source, time_delay, line_impedance, r_source, r_termination, iterations=None,
                print_reflection_coeffs=False, print_steady_state=False, print_initial_voltage=False,
                max_iterations=25, tolerance=0.01):
    # initial voltage
    v_initial = line_impedance * v_source / (line_impedance + r_source)

    # steady state voltage
    v_steady = r_termination * v_source / (r_source + r_termination)

    # compute reflection coefficients
    rho_near = (r_source - line_impedance) / (r_source + line_impedance)
    if r_termination not in (float('inf'), np.inf):
        rho_far = (r_termination - line_impedance) / (r_termination + line_impedance)

    # check for special case:
    if r_termination in (float('inf'), np.inf):
        rho_far = 1
        v_steady = v_source

    if r_termination == 0:
        rho_far = -1
        v_steady = 0

    # print coefficients
    if print_reflection_coeffs:
        print(f'Reflection coefficients:\n\t(rho_near, rho_far) = {(round(rho_near, 4), round(rho_far, 4))}.')

    if print_steady_state:
        print(f'Steady state voltages:\n\tv_steady ='
              f' {round(v_steady, 4)} [V]')

    if print_initial_voltage:
        print(f'Initial voltages:\n\tv_initial ='
              f' {round(v_initial, 4)} [V]')

    # initial values
    bounce_number = 0
    v_near_end = [0, 0, v_initial]
    v_far_end = [0, 0, 0]
    time_points = [-time_delay, 0, 0]

    # back and forth reflections
    forward_reflection_voltage = v_initial
    backward_reflection_voltage = 0

    while True:
        # in each iteration we append time twice and the last voltage value to get the sharp edges.
        bounce_number += 1  # start at 1, as 0'th bounce is given by starting conditions
        time_now = bounce_number * time_delay
        for i in range(2):
            time_points.append(time_now)  # keep time trace updated

        # ---------------------------------------------------------------------
        if not is_even(bounce_number):  # this is the far end case
            for i in range(2):
                v_near_end.append(v_near_end[-1])  # no change in near end yet, append old voltage

            # update backward reflection
            backward_reflection_voltage = rho_far * forward_reflection_voltage

            # update far end voltage by adding sum of backward and forward reflection
            v_far_end.append(v_far_end[-1])
            v_far_end.append(v_far_end[-1] + forward_reflection_voltage + backward_reflection_voltage)

        # ---------------------------------------------------------------------
        if is_even(bounce_number):  # this is the near end case
            for i in range(2):
                v_far_end.append(v_far_end[-1])  # no change in far end yet, append old voltage

            # update forward reflection
            forward_reflection_voltage = rho_near * backward_reflection_voltage

            # update near end voltage by adding sum of backward and forward reflection
            v_near_end.append(v_near_end[-1])
            v_near_end.append(v_near_end[-1] + backward_reflection_voltage + forward_reflection_voltage)

        # ---------------------------------------------------------------------
        # Check for breaking condition
        if iterations:  # if an iterations limit is given, only break if it is reached
            if bounce_number == iterations:
                v_far_end.append(v_far_end[-1])
                v_near_end.append(v_near_end[-1])
                time_points.append(time_now + time_delay)
                return time_points, v_near_end, v_far_end

        if not iterations:
            break_condition_near = np.abs((v_near_end[-1] - v_steady)) <= (tolerance * v_steady) \
                                   + tolerance * v_source * int(v_steady == 0)
            # When steady state is zero add some slack
            break_condition_far = np.abs((v_far_end[-1] - v_steady)) <= tolerance * v_steady \
                                  + tolerance * v_source * int(v_steady == 0)

            if (break_condition_near and break_condition_far) or bounce_number == max_iterations:
                # break loop if within 1 percent of steady state or max iterations reached
                v_far_end.append(v_far_end[-1])
                v_near_end.append(v_near_end[-1])
                time_points.append(time_now + time_delay)
                return time_points, v_near_end, v_far_end
