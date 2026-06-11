import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent blocking
import matplotlib.pyplot as plt

# Now run the original script
exec(open('CFD_code_square.py').read())

# Save instead of showing
plt.savefig('cfd_output.png', dpi=100, bbox_inches='tight')
print("Plot saved to cfd_output.png")
