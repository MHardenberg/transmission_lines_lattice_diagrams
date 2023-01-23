from queue import PriorityQueue
import numpy as np


def run_diagram(v_source, time_delays_tuple, line_impedances_tuple, r_source, r_termination,
                print_reflection_coeffs=True, print_steady_state=True, print_initial_voltage=True, tolerance=0.001):
    time_delay0, time_delay1 = time_delays_tuple
    line_impedance0, line_impedance1 = line_impedances_tuple

    # initial voltage
    v_initial = line_impedance0 * v_source / (line_impedance0 + r_source)

    # steady state voltage
    v_steady = r_termination * v_source / (r_source + r_termination)

    # compute reflection coefficients
    rho_a = (r_source - line_impedance0) / (r_source + line_impedance0)
    rho_b = (line_impedance1 - line_impedance0) / (line_impedance0 + line_impedance1)
    rho_c = (line_impedance0 - line_impedance1) / (line_impedance0 + line_impedance1)
    if r_termination not in (float('inf'), np.inf):
        rho_d = (r_termination - line_impedance1) / (r_termination + line_impedance1)

    # compute transmission coefficients
    t_b = 1 + rho_b
    t_c = 1 + rho_c

    # check for special case:
    if r_termination in (float('inf'), np.inf):
        v_steady = v_source
        rho_d = 1

    if r_termination == 0:
        v_steady = 0
        rho_d = -1

        # print coefficients
    if print_reflection_coeffs:
        print(f'Reflection coefficients:\n\trho_abcd = {(rho_a, rho_b, rho_c, rho_d)}.')

    if print_steady_state:
        print(f'Steady state voltages:\n\tv_steady ='
              f' {round(v_steady, 4)} [V]')

    if print_initial_voltage:
        print(f'Initial voltages:\n\tv_initial ='
              f' {round(v_initial, 4)} [V]')

    # beams are represented as tuples containing time and abcd voltages.
    # a is the near end, b is the left central, c right central and d is the far end interface
    v_a, v_b, v_c, v_d = 0, v_initial, 0, 0  # first beam we consider is going from A to B
    time = time_delay0  # this is the first wave going from A to B through the left material
    beam = (time, np.array([v_a, v_b, v_c, v_d]))

    # Voltage levels
    # These are lists containing the voltage levels at different time steps
    near_end_level, central_level, far_end_level = [[-time_delay0, 0, 0], [0, 0, v_initial]], \
        [[-time_delay0, 0], [0, 0]], [[-time_delay0, 0], [0, 0]]

    # initiate a priority queue
    beam_queue = PriorityQueue()
    beam_queue.put(beam)

    while not beam_queue.empty():
        time, v_abcd = beam_queue.get()
        v_a, v_b, v_c, v_d = v_abcd

        if max(np.abs(v_abcd)) <= max(tolerance, tolerance * v_steady):
            # if the reflection has become so weak, we don't treat it further
            continue

        if v_a != 0:  # meaning a wave going into A
            # add in and outgoing waves to last value
            for _ in range(2):
                near_end_level[0].append(time)
            near_end_level[1].append(near_end_level[1][-1])
            near_end_level[1].append(near_end_level[1][-1] + v_a * (1 + rho_a))

            time += time_delay0  # as we propagate through the left line
            beam = (time, (0, v_a * rho_a, 0, 0))
            beam_queue.put(beam)
            continue

        if v_b != 0:  # meaning a wave going into B
            # add in and outgoing waves to last value
            for _ in range(2):
                central_level[0].append(time)
            central_level[1].append(central_level[1][-1])
            central_level[1].append(central_level[1][-1] + v_b * (1 + rho_b))
            # we generate a new beam from the reflection:
            new_beam = (time + time_delay0, (v_b * rho_b, 0, 0, 0))  # reflected beam travels through left line
            beam_queue.put(new_beam)

            # handle current beam
            time += time_delay1  # as we propagate through the right line
            beam = (time, (0, 0, 0, v_b * t_b))  # transmit beam to D
            beam_queue.put(beam)
            continue

        if v_c != 0:  # meaning a wave going into C
            # add in and outgoing waves to last value
            for _ in range(2):
                central_level[0].append(time)
            central_level[1].append(central_level[1][-1])
            central_level[1].append(central_level[1][-1] + v_c * (1 + rho_c))

            # we generate a new beam from the reflection:
            new_beam = (time + time_delay1, (0, 0, 0, v_c * rho_c))  # reflected beam travels through right line
            beam_queue.put(new_beam)

            # handle current beam
            time += time_delay0  # as we propagate through the left line
            beam = (time, (v_c * t_c, 0, 0, 0))  # transmit beam to A
            beam_queue.put(beam)
            continue

        if v_d != 0:  # meaning a wave going into D
            # add in and outgoing waves to last value
            for _ in range(2):
                far_end_level[0].append(time)
            far_end_level[1].append(far_end_level[1][-1])
            far_end_level[1].append(far_end_level[1][-1] + v_d * (1 + rho_d))

            time += time_delay1  # as we propagate through the left line
            beam = (time, (0, 0, v_d * rho_d, 0))  # send beam back to C
            beam_queue.put(beam)
            continue

    time += max(time_delay0, time_delay1)  # append final values to flatten out
    near_end_level[0].append(time)
    near_end_level[1].append(near_end_level[1][-1])
    central_level[0].append(time)
    central_level[1].append(central_level[1][-1])
    far_end_level[0].append(time)
    far_end_level[1].append(far_end_level[1][-1])
    return near_end_level, central_level, far_end_level
