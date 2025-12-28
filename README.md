# FROG_zaber_OceanOptics
FROG pulse analysis for high repitition rate laser systems using a zaber stage  and Ocean Optics HR4000 spectrometer.

the main aquisition file is zaber_frog_trial_3.py

set the proper parameter values and run it.

Since the it uses non-collinear SHG geomtery, the best overlap of the two pulses can only be found by observing the highest intensity green signal.

Once the highest intensity green signal has been established, the corresponding value of the delay stage can be marked as t=0.

this can be done by using the zaber software or the GUI version of this code.

the delay stage value goes into the approx midpoint parameter.

Check the port name of trhe stage and change the variable value accordingly.

0.0001mm is the minimum steo size possible for the stage.

scan distance has to be fixed as per the approximate pulse width.

Spec integration time is the time for which each spectral aquisition is integrated. Has to be adjusted according to the intensityu of the SHG signal.

total travel range is a property of the stage and does not vary for the same stage. It is the farthest distance that the stage can travel.

