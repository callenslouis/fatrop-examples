import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def show_skeleton(ax, q):
    # segment lengths
    L_thigh = 0.44
    L_shank = 0.45

    # Pelvis
    pelvis_tx = q[1]
    pelvis_ty = q[2]
    pelvis_tilt = q[0]
    
    # Joint angles
    hip_r_angle = q[3]
    hip_l_angle = q[4]
    knee_r_angle = q[5]
    knee_l_angle = q[6]
    ankle_r_angle = q[7]
    ankle_l_angle = q[8]

    # Joint positions
    pelvis_pos = np.array([pelvis_tx, pelvis_ty])
    
    # Hip positions (relative to pelvis)
    hip_r_pos = pelvis_pos
    hip_l_pos = pelvis_pos
    
    # Knee positions
    knee_r_pos = hip_r_pos + L_thigh * np.array([np.sin(pelvis_tilt + hip_r_angle), -np.cos(pelvis_tilt + hip_r_angle)])
    knee_l_pos = hip_l_pos + L_thigh * np.array([np.sin(pelvis_tilt + hip_l_angle), -np.cos(pelvis_tilt + hip_l_angle)])
    
    # Ankle positions
    ankle_r_pos = knee_r_pos + L_shank * np.array([np.sin(pelvis_tilt + hip_r_angle + knee_r_angle), -np.cos(pelvis_tilt + hip_r_angle + knee_r_angle)])
    ankle_l_pos = knee_l_pos + L_shank * np.array([np.sin(pelvis_tilt + hip_l_angle + knee_l_angle), -np.cos(pelvis_tilt + hip_l_angle + knee_l_angle)])

    # Plot skeleton
    ax.plot([hip_r_pos[0], knee_r_pos[0]], [hip_r_pos[1], knee_r_pos[1]], 'r-') # Right thigh
    ax.plot([hip_l_pos[0], knee_l_pos[0]], [hip_l_pos[1], knee_l_pos[1]], 'b-') # Left thigh
    ax.plot([knee_r_pos[0], ankle_r_pos[0]], [knee_r_pos[1], ankle_r_pos[1]], 'r-') # Right shank
    ax.plot([knee_l_pos[0], ankle_l_pos[0]], [knee_l_pos[1], ankle_l_pos[1]], 'b-') # Left shank

    # Plot joints
    ax.plot(pelvis_pos[0], pelvis_pos[1], 'ko', markersize=10, label='Pelvis')
    ax.plot(knee_r_pos[0], knee_r_pos[1], 'ro', markersize=5, label='Right Knee')
    ax.plot(knee_l_pos[0], knee_l_pos[1], 'bo', markersize=5, label='Left Knee')
    ax.plot(ankle_r_pos[0], ankle_r_pos[1], 'ro', markersize=5, label='Right Ankle')
    ax.plot(ankle_l_pos[0], ankle_l_pos[1], 'bo', markersize=5, label='Left Ankle')
    
    # ax.legend()
    return pelvis_pos[0] # return forward position of the pelvis for setting x limits in animation


def visualize_gait_cycle(xx_sol, n_coords, N, dt, problem_type):
    q_mesh = xx_sol[:n_coords, :]

    step_time = 1.0 / 0.9 / 2.0
    step_length = 1.2 * step_time
        
    # joints:
    # pelvis_tilt, pelvis_tx, pelvis_ty, hip_r, hip_l, knee_r, knee_l,
    # ankle_r, ankle_l
    # k = 0         k = N
    # ------------------------
    # pelvis_tx     pelvis_tx
    # pelvis_ty     pelvis_ty
    # hip_r         hip_l
    # hip_l         hip_r
    # knee_r        knee_l
    # knee_l        knee_r
    second_period_copy_idxs = [0, 1, 2, 4, 3, 6, 5, 8, 7] # swap left and right leg joints for the second half of the gait cycle

    # Construct full cycle from solution for half cycle
    q_GC = np.hstack([q_mesh[:, :], q_mesh[second_period_copy_idxs,1:]])
    q_GC[1, N+1:] = q_GC[1, N+1:] + step_length

    t_GC = np.linspace(0, 2*dt*N, 2*N+1)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Walking Gait Simulation Results')

    # Torso Angle
    axs[0, 0].plot(t_GC, q_GC[0, :] * 180 / np.pi)
    axs[0, 0].set_title('Torso Angle')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Angle (°)')

    # Forward Position
    axs[0, 1].plot(t_GC, q_GC[1, :])
    axs[0, 1].set_title('Forward Position')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Position (m)')

    # Vertical Position
    axs[0, 2].plot(t_GC, q_GC[2, :])
    axs[0, 2].set_title('Vertical Position')
    axs[0, 2].set_xlabel('Time (s)')
    axs[0, 2].set_ylabel('Position (m)')

    # Hip Angle
    axs[1, 0].plot(t_GC, q_GC[4, :] * 180 / np.pi)
    axs[1, 0].set_title('Hip Angle')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Angle (°)')

    # Knee Angle
    axs[1, 1].plot(t_GC, q_GC[6, :] * 180 / np.pi)
    axs[1, 1].set_title('Knee Angle')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Angle (°)')

    # Ankle Angle
    axs[1, 2].plot(t_GC, q_GC[8, :] * 180 / np.pi)
    axs[1, 2].set_title('Ankle Angle')
    axs[1, 2].set_xlabel('Time (s)')
    axs[1, 2].set_ylabel('Angle (°)')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    # animate skeleton
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def update(frame):
        ax.clear()
        x_pos = show_skeleton(ax, q_GC[:, frame])
        ax.set_xlim(x_pos - 0.5, x_pos + 1)
        ax.set_ylim(-0.5, 1.5)
        ax.axhline(0, color='k', linestyle='-', lw=2)
        # show small lines at distance half steplength to highlight motion
        for k in range(-5, 5):
            ax.plot([k*step_length/2, k*step_length/2], 
                    [0, -0.01], color='k', linestyle='-', alpha=1)
        
        ax.set_aspect('equal')
        ax.set_xlabel('Forward Position (m)')
        ax.set_ylabel('Vertical Position (m)')


    nb_frames = q_GC.shape[1]
    fps = 1/dt
    ani = FuncAnimation(fig, update, frames=nb_frames, repeat=True)
    ani.save(f"visualization_output/walking_gait_skeleton_{problem_type}.gif", writer='pillow', fps=30)