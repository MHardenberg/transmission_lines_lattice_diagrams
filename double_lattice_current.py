from queue import PriorityQueue
import numpy as np


def run_diagram(v_source, time_delays_tuple, line_impedances_tuple, r_source, r_termination,
                        print_reflection_coeffs=True, print_initial_current=True, tolerance=0.001):
    time_delay0, time_delay1 = time_delays_tuple
    line_impedance0, line_impedance1 = line_impedances_tuple

    # initial current
    i_initial = v_source / (r_termination + r_source)

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
        rho_d = 1

    if r_termination == 0:
        rho_d = -1

        # print coefficients
    if print_reflection_coeffs:
        print(f'Reflection coefficients:\n\trho_abcd = {(rho_a, rho_b, rho_c, rho_d)}.')

    if print_initial_current:
        print(f'Initial current:\n\ti_initial ='
              f' {round(i_initial, 4)} [A]')

    # beams are represented as tuples containing time and abcd voltages.
    # a is the near end, b is the left central, c right central and d is the far end interface
    i_a, i_b, i_c, i_d = 0, i_initial, 0, 0  # first beam we consider is going from A to B
    time = time_delay0  # this is the first wave going from A to B through the left material
    beam = (time, (i_a, i_b, i_c, i_d))

    # Voltage levels
    # These are lists containing the voltage levels at different time steps
    near_end_level, central_level, far_end_level = [[-time_delay0, 0, 0], [0, 0, i_initial]], \
        [[-time_delay0, 0], [0, 0]], [[-time_delay0, 0], [0, 0]]

    # initiate a priority queue
    beam_queue = PriorityQueue()
    beam_queue.put(beam)

    while not beam_queue.empty():
        time, i_abcd = beam_queue.get()
        i_a, i_b, i_c, i_d = i_abcd
        if max(np.abs(i_abcd)) <= max(tolerance, tolerance * i_initial):
            # if the reflection has become so weak, we don't treat it further
            continue

        if i_a != 0:  # meaning a wave going into A
            # add in and outgoing waves to last value
            for _ in range(2):
                near_end_level[0].append(time)
            near_end_level[1].append(near_end_level[1][-1])
            near_end_level[1].append(near_end_level[1][-1] + i_a * (1 + rho_a))

            time += time_delay0  # as we propagate through the left line
            beam = (time, (0, i_a * rho_a, 0, 0))
            beam_queue.put(beam)
            continue

        if i_b != 0:  # meaning a wave going into B
            # add in and outgoing waves to last value
            for _ in range(2):
                central_level[0].append(time)
            central_level[1].append(central_level[1][-1])
            central_level[1].append(central_level[1][-1] + i_b * (1 + rho_b))
            # we generate a new beam from the reflection:
            new_beam = (time + time_delay0, (i_b * rho_b, 0, 0, 0))  # reflected beam travels through left line
            beam_queue.put(new_beam)

            # handle current beam
            time += time_delay1  # as we propagate through the right line
            beam = (time, (0, 0, 0, i_b * t_b))  # transmit beam to D
            beam_queue.put(beam)
            continue

        if i_c != 0:  # meaning a wave going into C
            # add in and outgoing waves to last value
            for _ in range(2):
                central_level[0].append(time)
            central_level[1].append(central_level[1][-1])
            central_level[1].append(central_level[1][-1] + i_c * (1 + rho_c))

            # we generate a new beam from the reflection:
            new_beam = (time + time_delay1, (0, 0, 0, i_c * rho_c))  # reflected beam travels through right line
            beam_queue.put(new_beam)

            # handle current beam
            time += time_delay0  # as we propagate through the left line
            beam = (time, (i_c * t_c, 0, 0, 0))  # transmit beam to A
            beam_queue.put(beam)
            continue

        if i_d != 0:  # meaning a wave going into D
            # add in and outgoing waves to last value
            for _ in range(2):
                far_end_level[0].append(time)
            far_end_level[1].append(far_end_level[1][-1])
            far_end_level[1].append(far_end_level[1][-1] + i_d * (1 + rho_d))

            time += time_delay1  # as we propagate through the left line
            beam = (time, (0, 0, i_d * rho_d, 0))  # send beam back to C
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
