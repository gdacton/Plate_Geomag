# Module plate_geomag.py: A module that contains functions for plate tectonic, geographic, geomagnetic, and paleomagnetic applications.
# Angles are in degrees unless otherwise specified. The original function required input in radians.
# 
# Required libraries: numpy, pandas

############################################################################################################
import numpy as np
import pandas as pd
############################################################################################################


"""
Design philosophy:
- Most geometric operations operate on DataFrames column-wise.
- Some generators (fisherdis, smcir) return new DataFrames.
- Summary statistics may return scalars.
"""


# Many of these functions came from the lplate library, written by geoscience students and faculty at Northwestern University in the 1980s.

# Functions in this module:
# adjust_az:       adjusts azimuth (e.g., magnetic declination) values to fall between 0 and 360 degrees
# angdis:          finds the angular distance between two points
# antipode:        finds the antipodes of points on the globe for a dataframe
# aztran:          finds the azimuth of a transform given a pole
# bingham_stats:   computes Bingham (axial) statistics for unit vectors defined by (inc/lat, dec/lon), with bootstrap confidence ellipse
# car3sp:          converts cartesian coordinates to spherical coordinates
# carsph:          converts cartesian coordinates to spherical coordinates for unit radius
# check_columns:   utility function to check for required columns
# ellipse:         generates the latitudes and longitudes for an error ellipse about a pole (e.g. the 95% confidence ellipse for a paleomagnetic pole)
# fisherdis:       generates a DataFrame with N Fisher-distributed deviates around a mean direction or pole
# Fisher_stats:    function to compute Fisher statistics for a dataframe
# incdec:          computes inclination and declination (paleomagnetic directions) at a site from a pole (VGP)
# intersect:       finds the intersection of two great circles on a unit sphere (NOT IMPLETMENTED)
# invert_reversed: invert data with negative inclination to have common polarity
# locate:          finds a point on the globe at given distance and azimuth from an initial point
# polaz:           finds pole given point and azimuth of great circle
# polpts:          finds the pole for two points on a great circle; the two points are on the equator of the pole
# rot1:            rotate 'angle' radians about axis #1
# rot2:            rotate 'angle' radians about axis #2
# rot3:            rotate 'angle' radians about axis #3
# rotate:          rotates a point about an Euler pole using a rotation matrix; results should be identical to rotp
# rotaz:           finds the new location and azimuth after a rotation
# rotp:            rotates w degrees about an arbitrary pole using rot1, rot2, and rot3; results should be identical to rotate
# smcir:           generates the latitudes and longitudes for a small circle about a pole, like an A95 circle.
# sphcar:          changes spherical coordinates to cartesian coordinates for unit radius
# vector_mean:     finds the vector mean for a set of inclinations and declinations or latitudes and longitudes
# vgp:             calculates Virtual Geomagnetic Poles (VGPs) from inclination and declination
#
# gmfact:         function to calculate geometrical factor to convert angular velocity
#                 to rms velocity given certain integrals over the surface of the
#                 plate or continent and the lat and lon of the angular velocity
#                 vector (i.e. fixed euler pole)
#                 gmfact.r is written to work in radians only, just like these other routines.
#                 It has not, however, been tested.
# rotaz:          finds azimuth after rotation from old azimuth
# sumrot:         finds equivalent of two successive rotations
# veloc1:         finds velocity owing to rotation

def adjust_az(df, az_name, az_min, az_max):
    """
    Adjust azimuth (e.g., declination) values in the specified column to fall between az_min and az_max degrees.
    This uses modulo arithmetic to ensure that the values are between az_min (inclusive) and az_max (exclusive).
    
    az_name:  name of the column containing azimuth values
    az_min:   minimum azimuth value (e.g., 0.0); if the value is lower, it is adjusted by adding 360.
    az_max:   maximum azimuth value (e.g., 360.0); if the value is greater or equal, it is adjusted by subtracting 360.
    
    Example call
    df = adjust_az(df, 'Declination', -90.0, 270.0)
    """
    # print('In function adjust_az')
    dfn = df.copy()
    
    # Old loop-based approach, which would not catch cases where az_max - az_min > 360
    # for i in dfn.index:
    #     value = dfn.loc[i, az_name]
    #     if value < az_min:
    #         value = value + 360.0
    #     elif value >= az_max:
    #         value = value - 360.0
    #     dfn.loc[i, az_name] = value
    
    # Alternative vectorized approach
    span = az_max - az_min
    dfn[az_name] = ((dfn[az_name] - az_min) % span) + az_min
    return dfn



############################################################################################################
def angdis(df, Lat1, Lon1, Lat2, Lon2, AngDis):
    """
    Function angdis computes the angular distance between two points on a sphere.
    The angular distance, latitude, and longitude are assumed to be in degrees.
    """
    # Example call
    # df = angdis(df, 'Site Lat', 'Site Lon', 'Pole Lat', 'Pole Lon', 'Angular Distance')
    
    # Column names in call to function must be in the dataframe
    # if not check_columns(df, [Lat1, Lon1, Lat2, Lon2, AngDis]):
    #     return df

    dfn = df.copy()
    dfn1 = sphcar(dfn, Lat1, Lon1, 'x', 'y', 'z')
    dfn2 = sphcar(dfn, Lat2, Lon2, 'x', 'y', 'z')

    # Dot product of unit vectors
    check = dfn1['x'] * dfn2['x'] + dfn1['y'] * dfn2['y'] + dfn1['z'] * dfn2['z']

    # Numerical safety:
    # - force values extremely close to ±1 to exactly ±1 (prevents tiny arccos angles for identical points)
    # - only warn if we're meaningfully outside [-1, 1]
    eps = 1e-14

    if check.max() > 1.0 + 1e-12 or check.min() < -1.0 - 1e-12:
        print(f"Warning: check outside [-1,1] beyond tolerance. max={check.max()}, min={check.min()}")

    check = check.where(check < 1.0 - eps, 1.0)
    check = check.where(check > -1.0 + eps, -1.0)

    dfn[AngDis] = np.degrees(np.arccos(check.clip(-1.0, 1.0)))
    return dfn


############################################################################################
def antipode(df, inc, dec, inc_inverted, dec_inverted):
    """
    Invert all the data to find the antipodes.
    
    Example call
    df = antipode (df, 'Inclination', 'Declination', 'Inc_Inverted', 'Dec_Inverted')
    """
    dfn = df.copy()
    dfn[inc_inverted] = -1 * dfn[inc]
    dfn[dec_inverted] = (dfn[dec] + 180) % 360
    return dfn

############################################################################################################
def aztran(df, pole_lat, pole_lon, lat, lon, azimuth):
    """
    Function aztran finds the azimuth of a transform given a pole.
    The input and output latitudes, longitudes, and azimuth are in degrees.
    The azimuth is measured **clockwise from north**.

    Args:
        df (DataFrame): Input dataframe containing the transform coordinates.
        pole_lat (str): Column name for latitude of the pole.
        pole_lon (str): Column name for longitude of the pole.
        lat (str): Column name for latitude of the transform point.
        lon (str): Column name for longitude of the transform point.
        azimuth (str): Column name for output transform azimuth.

    Returns:
        DataFrame: A copy of the input dataframe with the computed transform azimuth.

    Example usage:
    ```python
    df = aztran(df, 'Pole_Lat', 'Pole_Lon', 'Lat', 'Lon', 'Azimuth')
    ```
    """

    dfn = df.copy()

    # Compute 180 - lon and store it in a new column
    dfn['rotation1'] = 180.0 - dfn[lon]

    # Step 1: Rotate the pole around the z-axis (rot3)
    dfn = rot3(dfn, pole_lat, pole_lon, 'rotation1', 'tlat1_aztran', 'tlon1_aztran')

    # Compute 90 - lat and store it in a new column
    dfn['rotation2'] = 90.0 - dfn[lat]

    # Step 2: Rotate around the y-axis (rot2)
    dfn = rot2(dfn, 'tlat1_aztran', 'tlon1_aztran', 'rotation2', 'tlat2_aztran', 'tlon2_aztran')

    # Step 3: Compute azimuth (CW from North)
    dfn[azimuth] = 90.0 - dfn['tlon2_aztran']

    # Drop intermediate transformation columns
    dfn = dfn.drop(columns=['rotation1', 'tlat1_aztran', 'tlon1_aztran', 'rotation2', 'tlat2_aztran', 'tlon2_aztran'])
    
    dfn = adjust_az(dfn, azimuth, 0.0, 360.0)

    return dfn

############################################################################################################
def bingham_stats(df, inc_lat, dec_lon, weight_col=None,
                  conf=0.95, n_boot=5000, seed=0,
                  return_bootstrap=False):
    
    """
    Compute Bingham (axial) statistics for a set of unit vectors defined by
    inclination/latitude and declination/longitude, and estimate an elliptical
    confidence region about the principal axis using bootstrap resampling.

    The confidence region is represented as an ellipse in the tangent plane
    at the mean axis. The ellipse is described by its semi-major and semi-minor
    axes (in degrees) and by the azimuth of the major axis measured clockwise
    from geographic north at the mean axis.

    All angles are in **degrees**, except where noted internally.

    Args:
        df (DataFrame): Input dataframe containing directional data.
        inc_lat (str): Column name for inclination or latitude (degrees).
        dec_lon (str): Column name for declination or longitude (degrees).
        weight_col (str or None): Optional column of weights used in forming the
            orientation matrix and bootstrap resampling probabilities.
        conf (float): Confidence level for the ellipse (e.g., 0.95).
        n_boot (int): Number of bootstrap replicates.
        seed (int or None): Random number generator seed.
        return_bootstrap (bool): If True, return bootstrap mean axes and their
            tangent-plane coordinates.

    Returns:
        dict: Dictionary containing Bingham statistics with keys:
            - N: Number of vectors used.
            - mean_axis_inc_lat: Inclination/latitude of the mean axis (degrees).
            - mean_axis_dec_lon: Declination/longitude of the mean axis (degrees).
            - tau: Eigenvalues of the orientation matrix (descending order).
            - eigenvectors: Corresponding eigenvectors (columns).
            - ellipse: Dictionary with keys:
                - conf: Confidence level.
                - a_deg: Semi-major axis length (degrees).
                - b_deg: Semi-minor axis length (degrees).
                - az_deg: Azimuth of the major axis (degrees, clockwise from north).
                - note: Description of ellipse construction.
            - bootstrap_df (optional): DataFrame of bootstrap mean axes and
              tangent-plane coordinates.

    Example usage:
    ```python
    result = bingham_stats(df, 'Inclination', 'Declination',
                           weight_col='Weight', n_boot=1000)
    print(result['mean_axis_inc_lat'], result['mean_axis_dec_lon'])
    print(result['ellipse']['a_deg'], result['ellipse']['b_deg'],result['ellipse']['az_deg'])
    ```

    Example with VGP latitude/longitude and bootstrap output:
    ```python
    result = bingham_stats(df, inc_lat='VGP_Lat', dec_lon='VGP_Lon',
                           conf=0.95, n_boot=10000, return_bootstrap=True)
    ```
    """
        
    # --- checks / clean ---
    for col in (inc_lat, dec_lon):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

    dfn = df[[inc_lat, dec_lon] + ([weight_col] if weight_col else [])].copy()
    dfn = dfn.dropna(subset=[inc_lat, dec_lon])
    if len(dfn) < 3:
        raise ValueError("Need at least 3 non-NaN vectors for Bingham statistics.")

    # --- unit vectors via sphcar ---
    dfn_xyz = sphcar(dfn, inc_lat, dec_lon, 'x', 'y', 'z')
    V = dfn_xyz[['x', 'y', 'z']].to_numpy(dtype=float)  # (N,3)
    N = V.shape[0]

    # Optional weights
    w = None
    p = None
    if weight_col is not None:
        if weight_col not in dfn.columns:
            raise ValueError(f"weight_col '{weight_col}' not found in df.")
        w = dfn[weight_col].to_numpy(dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if np.all(w == 0):
            w = None
        else:
            p = w / w.sum()

    # --- orientation/scatter matrix T = sum w_i v_i v_i^T ---
    if w is None:
        T = V.T @ V
    else:
        T = (V * w[:, None]).T @ V

    # --- principal axis from symmetric eigendecomposition ---
    evals, evecs = np.linalg.eigh(T)          # ascending
    idx = np.argsort(evals)[::-1]             # descending
    tau = evals[idx]
    evecs = evecs[:, idx]                     # columns are eigenvectors
    m = evecs[:, 0].copy()                    # principal axis (unit)
    m /= np.linalg.norm(m)

    # Axial convention: choose a stable sign
    # (z >= 0; if z ~ 0, y >= 0)
    if (m[2] < 0) or (abs(m[2]) < 1e-12 and m[1] < 0):
        m = -m
        evecs[:, 0] = -evecs[:, 0]

    # Convert mean axis back to (inc/dec) using carsph on a 1-row DF
    dfm = pd.DataFrame({'x': [m[0]], 'y': [m[1]], 'z': [m[2]]})
    dfm = carsph(dfm, 'x', 'y', 'z', '_inc_lat_mean', '_dec_lon_mean')
    mean_inc_lat = float(dfm.loc[0, '_inc_lat_mean'])
    mean_dec_lon = float(dfm.loc[0, '_dec_lon_mean'] % 360.0)

    # --- tangent basis at m using geographic North/East ---
    # We define azimuth clockwise from North:
    #   az=0 => major axis points to local North
    #   az=90 => points to local East
    eps = 1e-12
    h = np.hypot(m[0], m[1])  # horizontal magnitude

    if h < eps:
        # Mean axis is (numerically) at a pole: choose a stable arbitrary basis
        eE = np.array([0.0, 1.0, 0.0])  # "east"
        eN = np.array([1.0, 0.0, 0.0])  # "north"
    else:
        # Local East at (lat,lon) implied by m
        eE = np.array([-m[1] / h, m[0] / h, 0.0])
        eE /= np.linalg.norm(eE)

        # Local North is perpendicular to East and Up(m): N = Up x East
        eN = np.cross(m, eE)
        eN /= np.linalg.norm(eN)


    # --- bootstrap axes and map to tangent plane via log map ---
    rng = np.random.default_rng(seed)
    boot_axes = np.empty((n_boot, 3), dtype=float)
    xy = np.empty((n_boot, 2), dtype=float)   # tangent coords in radians

    for b in range(n_boot):
        idxb = rng.choice(N, size=N, replace=True, p=p)
        Vb = V[idxb, :]
        if w is None:
            Tb = Vb.T @ Vb
        else:
            wb = w[idxb]
            Tb = (Vb * wb[:, None]).T @ Vb

        eb, vb = np.linalg.eigh(Tb)
        ib = np.argsort(eb)[::-1]
        mb = vb[:, ib[0]]
        mb = mb / np.linalg.norm(mb)

        # axial sign: align with m
        if np.dot(mb, m) < 0:
            mb = -mb

        boot_axes[b, :] = mb

        # log map at m:
        c = np.clip(np.dot(m, mb), -1.0, 1.0)
        ang = np.arccos(c)  # radians
        if ang < 1e-14:
            xy[b, :] = 0.0
        else:
            d = mb - c * m
            dn = np.linalg.norm(d)
            if dn < 1e-14:
                xy[b, :] = 0.0
            else:
                direction = d / dn
                # x = North component, y = East component (both in radians)
                xy[b, 0] = ang * np.dot(direction, eN)
                xy[b, 1] = ang * np.dot(direction, eE)

    # covariance in tangent plane (2D)
    C = np.cov(xy.T, bias=False)
    eval2, evec2 = np.linalg.eigh(C)          # ascending
    i2 = np.argsort(eval2)[::-1]
    eval2 = eval2[i2]
    evec2 = evec2[:, i2]                      # columns

    # For df=2, chi-square quantile has closed form: q = -2 ln(1-conf)
    if not (0.0 < conf < 1.0):
        raise ValueError("conf must be between 0 and 1.")
    chi2_q = -2.0 * np.log(1.0 - conf)

    # semi-axes in radians in tangent plane
    a = np.sqrt(max(eval2[0], 0.0) * chi2_q)
    b = np.sqrt(max(eval2[1], 0.0) * chi2_q)

    # evec2[:,0] is the major-axis direction in the (North, East) tangent basis
    # az = atan2(East, North)
    az = (np.degrees(np.arctan2(evec2[1, 0], evec2[0, 0])) % 360.0)

    out = {
        "N": int(N),
        "mean_axis_inc_lat": mean_inc_lat,
        "mean_axis_dec_lon": mean_dec_lon,
        "tau": tau.astype(float),
        "eigenvectors": evecs.astype(float),  # columns correspond to tau
        "ellipse": {
            "conf": float(conf),
            "a_deg": float(np.degrees(a)),
            "b_deg": float(np.degrees(b)),
            "az_deg": float(az),
            "note": "Ellipse is estimated in tangent plane at mean axis using bootstrap + 2D normal approx.",
        },
    }

    if return_bootstrap:
        # Convert bootstrap axes to inc/dec using carsph on a DF (vectorized)
        dfb = pd.DataFrame({'x': boot_axes[:, 0], 'y': boot_axes[:, 1], 'z': boot_axes[:, 2]})
        dfb = carsph(dfb, 'x', 'y', 'z', 'boot_inc', 'boot_dec')
        dfb['boot_dec'] = dfb['boot_dec'] % 360.0
        dfb['x_tan_deg'] = np.degrees(xy[:, 0])
        dfb['y_tan_deg'] = np.degrees(xy[:, 1])
        out["bootstrap_df"] = dfb[['boot_inc', 'boot_dec', 'x_tan_deg', 'y_tan_deg']].copy()

    return out


############################################################################################################
def car3sp(df, x, y, z, lat, lon, radius):
    """
    Function car3sp converts Cartesian coordinates to spherical coordinates (latitude, longitude, radius).
    The output latitude and longitude are in **degrees**.
    
    Args:
        df (DataFrame): Input dataframe containing Cartesian coordinates.
        x (str): Column name for x-coordinate.
        y (str): Column name for y-coordinate.
        z (str): Column name for z-coordinate.
        lat (str): Column name for output latitude (in degrees).
        lon (str): Column name for output longitude (in degrees).
        rad (str): Column name for output radius (length of vector).

    Returns:
        DataFrame: A copy of the input dataframe with new latitude, longitude (in degrees), and radius columns.

    Example usage:
    ```python
    df = car3sp(df, 'x', 'y', 'z', 'lat_deg', 'lon_deg', 'radius')
    ```
    """

    dfn = df.copy()
    # Compute the radius (length of the vector)
    dfn[radius] = np.sqrt(dfn[x]**2 + dfn[y]**2 + dfn[z]**2)

    # Handle the case where the radius is zero
    zero_radius_mask = dfn[radius] == 0.0
    dfn.loc[zero_radius_mask, lat] = 0.0
    dfn.loc[zero_radius_mask, lon] = 0.0

    # Convert nonzero radius values to latitude and longitude (in degrees)
    nonzero_mask = ~zero_radius_mask
    arg = (dfn.loc[nonzero_mask, z] / dfn.loc[nonzero_mask, radius]).clip(-1.0, 1.0)
    dfn.loc[nonzero_mask, lat] = np.degrees(np.arcsin(arg))
    dfn.loc[nonzero_mask, lon] = np.degrees(np.arctan2(dfn.loc[nonzero_mask, y], dfn.loc[nonzero_mask, x]))

    return dfn

############################################################################################################
def carsph(df, x, y, z, inc_lat, dec_lon):
    """
    Function carsph converts cartesian to spherical coordinates for a dataframe.
    The length of the vector is assumed to be 1.
    The spherical coordinates are assumed to be in degrees.
    The cartesian coordinates are in the same units as the vector, which is assumed to be 1.
    See https://en.wikipedia.org/wiki/Spherical_coordinate_system
    See https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_spherical_coordinates
    
    Example call
    df = carsph(df, 'xnorm', 'ynorm', 'znorm','Inc', 'Dec')
    """
    dfn = df.copy()
    # Compute the spherical coordinates for each x, y, z triple.
    dfn[inc_lat] = np.degrees(np.arcsin(dfn[z].clip(-1.0, 1.0)))
    dfn[dec_lon] = np.degrees(np.arctan2(dfn[y], dfn[x]))
    return dfn

############################################################################################################
def check_columns(df, required_cols):
    """
    Utility function to check for required columns
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Columns not found in DataFrame: {missing}")
        return False
    return True

############################################################################################################
# def ellipse(plat, plon, major_axis, minor_axis, az_major, num_points):
#     """
#     Generates latitude and longitude points for an ellipse centered on a given pole.

#     Parameters:
#         plat (float): Latitude of the pole in degrees.
#         plon (float): Longitude of the pole in degrees.
#         major (float): Major axis length in degrees.
#         minor (float): Minor axis length in degrees.
#         az_major (float): Azimuth of the major axis in degrees.
#         num_points (int): Number of points to generate for the ellipse (default  91).

#     Returns:
#         pd.DataFrame: A DataFrame with columns ['Latitude', 'Longitude'] representing the ellipse.
#     """

#     # Create ellipse in local coordinate system
#     angles = np.linspace(0, 360, num_points)  # Generate angles from 0° to 360°
#     print(f"Angles: {angles}")
#     print(f"Major axis: {major_axis}, Minor axis: {minor_axis}")
#     x = (major_axis) * np.cos(2*np.radians(angles))  # X in degrees
#     y = (minor_axis) * np.sin(2*np.radians(angles))  # Y in degrees
#     z = np.sqrt(1.0 - np.radians(x)**2 - np.radians(y)**2)  # Approximate unit sphere

#     # Convert to a DataFrame
#     df = pd.DataFrame({'x': x, 'y': y, 'z': z})

#     # Convert Cartesian coordinates to spherical (lat, lon)
#     df = carsph(df, 'x', 'y', 'z', 'vlat', 'vlon')
#     print(f"df after carsph(): {df.head()}")

#     # Rotate the ellipse using azimuth
#     df['templon'] = plon + 90.0
#     df['templat'] = 90.0 - plat
#     df['zero'] = 0.0
#     df['northpole'] = 90.0
#     df['az_major'] = az_major

#     # Rotate around the azimuth
#     df = rotp(df, 'northpole', 'zero', 'az_major', 'vlat', 'vlon', 'xlat', 'xlon')

#     # Rotate to the actual pole position
#     df = rotp(df, 'zero', 'templon', 'templat', 'xlat', 'xlon', 'Latitude', 'Longitude')

#     # Ensure longitudes are in the range [0, 360]
#     df['Longitude'] = df['Longitude'] % 360

#     # Keep only the necessary columns
#     df = df[['Latitude', 'Longitude']]

#     return df

# # # Example Usage
# # plat, plon = 30, 60  # Pole latitude and longitude
# # major, minor = 10, 5  # Major and minor axis lengths in degrees
# # az_major = 45  # Azimuth of major axis

# # df_ellipse =  ellipse(plat, plon, major_axis, minor_axis, az_major)

############################################################################################################
def fisherdis(alat_deg, alon_deg, kappa, N=1000, seed=None):
    """
    Generates a DataFrame with N Fisher-distributed deviates around a mean direction or pole.
    
    The output can be used to generate random directions
    around a mean direction (inclination and declination) or
    around a mean pole (pole latitude and longitude) with a Fisher dispersion parameter (KAPPA).
    The original algorithm is from the book "Statistical Analysis of Spherical Data
    by Fisher, Lewis, & Embleton (1987) page 59. 
    
    Parameters:
    - alat_deg: Mean latitude (degrees)
    - alon_deg: Mean longitude (degrees)
    - kappa: Fisher precision parameter
    - N: Number of random deviates
    - seed: Random seed for reproducibility

    Returns:
    - DataFrame with columns 'Latitude', 'Longitude', and 'Colatitude'
    
    Example usage:
    df_fisherian = fisherdis(90, 0, 100.0, N=1000, seed=43)
    print(df_fisherian.head(10)) # Display the first 10 rows of the DataFrame
    """
    rng = np.random.default_rng(seed)
    R1 = rng.random(N)
    R2 = rng.random(N)

    # Constants
    pi = np.pi
    lamda = np.exp(-2.0 * kappa)
    temp = -np.log(R1 * (1.0 - lamda) + lamda)
    colat_rad = 2.0 * np.arcsin(np.sqrt(temp / (2.0 * kappa)))

    # Deviates around the north pole (lat=90°, lon=0°)
    rlat_rad = pi / 2.0 - colat_rad
    rlon_rad = 2.0 * pi * R2

    # Convert to degrees
    df = pd.DataFrame({
        'Old_Lat': np.degrees(rlat_rad),
        'Old_Lon': np.degrees(rlon_rad),
        'Pole_Lat': 0.0,  # North pole
        'Pole_Lon': np.degrees((np.radians(alon_deg) + pi/2.0) % (2.0 * pi)),
        'Angle': np.degrees(pi/2.0 - np.radians(alat_deg))  # Rotation angle
    })

    # Use rotp (which relies on rot2 and rot3) to apply spherical rotation
    df = rotp(df, 'Pole_Lat', 'Pole_Lon', 'Angle', 'Old_Lat', 'Old_Lon', 'Latitude', 'Longitude')

    # Colatitudes in degrees
    df['Colatitude'] = np.degrees(colat_rad)

    return df[['Latitude', 'Longitude', 'Colatitude']]

############################################################################################################
def Fisher_stats(N, R):
    """
    Function to computer Fisher statistics for a dataframe given the number of data and the length of
    the resultant vector.
    
    N:  number of data points
    R:  length of resultant vector
    """
    if N < 2:
        print(f"Number of data points must be greater than 1.")
        return np.nan, np.nan, np.nan
    if R <= 0.0:
        print(f"Length of resultant vector must be greater than 0.")
        return np.nan, np.nan, np.nan
    if R > N:
        print("Warning: R > N (invalid).")
        return np.nan, np.nan, np.nan
    if R == N:
        return np.inf, 0.0, 0.0
    
    # Compute the Fisher statistics.
    k = (N - 1) / (N - R)
    a95 = 140/np.sqrt(k * N) # Approximation that gives similar values to equation below
    # print(f"A95 = {a95}")
    # Preferred general formula from Butler (Eqn 6.21) is given below with a clip check for arccos
    a95 = np.degrees(np.arccos(
        np.clip((1 - ((N - R)/ R) *  ((1 / 0.05)**(1 /(N - 1)) - 1)  ), -1, 1)
    ))
    # print(f"A95 = {a95}")
    
    
    # Another statistic, δ, which is often used as a measure of angular dispersion (and is often 
    # called the angular standard deviation) is given by δ = arccos(R/N)) (Butler, 1992), or can be
    # estimated by s = 81/K^2 (Butler, 1992).
    # dfm['s'] = 81.0 / np.sqrt(dfm['k'])
    angular_stdev = np.degrees(np.arccos(R/N))
    return k, a95, angular_stdev


############################################################################################################
def incdec(df, slat, slon, plat, plon, inc, dec, colat):
    # This function was converted to Python from Fortran 77 subroutine incdec.f.
    """
    Calculate inclination and declination at a site from a pole/VGP location.
    All angles are in degrees.

    Uses existing plate_geomag.py functions:
      - adjust_az() to normalize longitudes
      - angdis()   to compute colatitude
      - polpts()   to compute RH pole to the great circle (corrected version)
      - aztran()   to compute declination (CW from North)

    Example call:
    df = incdec(df,
                'Site_Lat', 'Site_Lon', 'VGP_Lat',  'VGP_Lon',
                'Inc_calc', 'Dec_calc', 'Colat_calc')
    """
    dfn = df.copy()

    # --- normalize longitudes to [0, 360) using existing utility ---
    dfn = adjust_az(dfn, slon, 0.0, 360.0)
    dfn = adjust_az(dfn, plon, 0.0, 360.0)

    # --- colatitude (degrees): angular distance between site and pole ---
    dfn = angdis(dfn, slat, slon, plat, plon, colat)

    # --- initialize outputs ---
    dfn[inc] = np.nan
    dfn[dec] = np.nan

    # --- special case: site == pole (Fortran branch) ---
    same_point = (dfn[slat] == dfn[plat]) & (dfn[slon] == dfn[plon])
    dfn.loc[same_point, inc] = 90.0
    dfn.loc[same_point, dec] = 0.0
    dfn.loc[same_point, colat] = 0.0

    # --- general case ---
    not_same = ~same_point
    if not_same.any():

        # Inclination from colatitude (piecewise Fortran form)
        colat_rad = np.radians(dfn[colat])

        mask_eq = not_same & (dfn[colat] == 90.0)
        dfn.loc[mask_eq, inc] = 0.0

        mask_lt = not_same & (dfn[colat] < 90.0)
        dfn.loc[mask_lt, inc] = np.degrees(
            np.arctan2(2.0, np.tan(colat_rad[mask_lt]))
        )

        mask_gt = not_same & (dfn[colat] > 90.0)
        dfn.loc[mask_gt, inc] = -np.degrees(
            np.arctan2(2.0, np.tan(np.pi - colat_rad[mask_gt]))
        )

        # Pole to the great circle that passes from site to the paleomagnetic pole
        dfn = polpts(dfn, slat, slon, plat, plon,
                     '_polat_incdec', '_polon_incdec')

        # Declination from aztran (CW from North)
        dfn = aztran(dfn, '_polat_incdec', '_polon_incdec',
                     slat, slon, dec)

        # --- Fortran override when slat == plat ---
        same_lat = not_same & (dfn[slat] == dfn[plat])
        if same_lat.any():

            # plon > slon
            mask_pos = same_lat & (dfn[plon] > dfn[slon])
            diff_pos = dfn.loc[mask_pos, plon] - dfn.loc[mask_pos, slon]
            dfn.loc[mask_pos & (diff_pos <= 180.0), dec] = 90.0
            dfn.loc[mask_pos & (diff_pos > 180.0), dec] = 270.0

            # plon < slon
            mask_neg = same_lat & (dfn[plon] < dfn[slon])
            diff_neg = dfn.loc[mask_neg, plon] - dfn.loc[mask_neg, slon]
            dfn.loc[mask_neg & (diff_neg <= -180.0), dec] = 90.0
            dfn.loc[mask_neg & (diff_neg > -180.0), dec] = 270.0

    # Ensure declination is between 0 and 360 degrees
    dfn = adjust_az(dfn, dec, 0.0, 360.0)

    # Simple sanity checks (same spirit as vgp())
    if dfn[colat].max() > 180.0 or dfn[colat].min() < 0.0:
        print("Warning: Colatitude is outside [0°, 180°].")
    if dfn[inc].max() > 90.0 or dfn[inc].min() < -90.0:
        print("Warning: Inclination is outside [-90°, 90°].")

    # Drop temporary columns
    dfn = dfn.drop(columns=['_polat_incdec', '_polon_incdec'], errors='ignore')

    return dfn



############################################################################################################
def invert_reversed(df, inc, dec, inc_inverted, dec_inverted):
    """
    Invert data with negative inclination to have common polarity, e.g., invert reversed 
    inclinations and declinations to normal polarity for nothern hemisphere data.
    
    Example call
    df = invert_reversed(df, 'Inclination', 'Declination', 'Inc_Inverted', 'Dec_Inverted')
    """
    df = adjust_az(df, dec, 0.0,360.0)
    dfn = df.copy()
    dfn[inc_inverted] = dfn[inc]
    dfn[dec_inverted] = dfn[dec]
    dfn.loc[dfn[inc] < 0, inc_inverted] = -1 * dfn[inc]
    dfn.loc[dfn[inc] < 0, dec_inverted] = (dfn[dec] + 180) % 360
    dfn = adjust_az(dfn, dec_inverted, 0.0,360.0)
    return dfn

############################################################################################################
def locate(df, lat1, lon1, angdis, az, lat2, lon2):
    """
    Finds a point on the globe at given distance and azimuth from an initial point.
    All angles are in **degrees**.
    
    Args:
        df (DataFrame): Input dataframe containing the reference points.
        lat1 (str): Column name for input latitude.
        lon1 (str): Column name for input longitude.
        angdis (str): Column name for angular distance from reference point.
        az (str): Column name for azimuth of desired point (CW from North).
        lat2 (str): Column name for output latitude.
        lon2 (str): Column name for output longitude.

    Returns:
        DataFrame: A copy of the input dataframe with computed (`lat2`, `lon2`).

    Example usage:
    ```python
    df = locate(df, 'Lat1', 'Lon1', 'AngDis', 'Azimuth', 'Lat2', 'Lon2')
    ```
    """
    dfn = df.copy()
    
    # Step 1: Convert angular distance and azimuth to needed rotation angles
    dfn['tlat_loc'] = 90.0 - dfn[angdis]  # Pi/2 - Angular distance
    dfn['tlon_loc'] = 180.0 - dfn[az]     # Pi - Azimuth
    dfn['trot_loc'] = 90.0 - dfn[lat1] # Pi/2 - Latitude
    dfn['trot1_loc'] = dfn[lon1]
    
    # print("\nInside locate(): After computation")
    # print(dfn.head())

    # Step 2: Rotate about y-axis using rot2
    dfn = rot2(dfn, 'tlat_loc', 'tlon_loc', 'trot_loc', 'tlat1_loc', 'tlon1_loc')

    # Step 3: Rotate about z-axis using rot3
    dfn = rot3(dfn, 'tlat1_loc', 'tlon1_loc', 'trot1_loc', lat2, lon2)

    # Normalize longitude to be within [0, 360)
    dfn[lon2] = dfn[lon2] % 360
    
    # print("\nInside locate(): After rot2 and rot3")
    # print("Final columns:", dfn.columns)
    # print(dfn.head())

    # Drop intermediate columns
    dfn = dfn.drop(columns=['tlat_loc', 'tlon_loc', 'tlat1_loc', 'tlon1_loc', 'trot_loc', 'trot1_loc'])

    return dfn


############################################################################################################
def polaz(df, lat, lon, az, pole_lat, pole_lon):
    """
    Function polaz finds a pole given a point and the azimuth of a great circle.
    The input and output latitudes, longitudes, and azimuth are in **degrees**.
    The azimuth is measured clockwise from north.

    Args:
        df (DataFrame): Input dataframe containing the point and azimuth.
        lat (str): Column name for latitude of the point.
        lon (str): Column name for longitude of the point.
        az (str): Column name for azimuth of the great circle.
        pole_lat (str): Column name for output pole latitude.
        pole_lon (str): Column name for output pole longitude.

    Returns:
        DataFrame: A copy of the input dataframe with new pole latitude and longitude columns.

    Example usage:
    ```python
    df = polaz(df, 'Lat', 'Lon', 'Azimuth', 'Pole_Lat', 'Pole_Lon')
    ```
    """

    dfn = df.copy()

    # Step 1: Perform first rotation (Rot2)
    dfn = rot2(dfn, 0.0, 90.0 - dfn[az], 90.0 - dfn[lat], 'tlat_polaz', 'tlon_polaz')

    # Step 2: Perform second rotation (Rot3)
    dfn = rot3(dfn, 'tlat_polaz', 'tlon_polaz', lon, pole_lat, pole_lon)
    
    # Drop intermediate columns
    dfn = dfn.drop(columns=['tlat_polaz', 'tlon_polaz'])

    return dfn

############################################################################################################
# OBSELETED VERSION OF polpts because ChatGPT caught a bug in it.
# def polpts(df, pt1_lat, pt1_lon, pt2_lat, pt2_lon, pole_lat, pole_lon):
#     """
#     Function polpts finds a pole given two points on the equator.
#     The input and output latitudes and longitudes are in **degrees**.

#     Args:
#         df (DataFrame): Input dataframe containing the two points.
#         pt1_lat (str): Column name for latitude of the first point.
#         pt1_lon (str): Column name for longitude of the first point.
#         pt2_lat (str): Column name for latitude of the second point.
#         pt2_lon (str): Column name for longitude of the second point.
#         pole_lat (str): Column name for output pole latitude.
#         pole_lon (str): Column name for output pole longitude.

#     Returns:
#         DataFrame: A copy of the input dataframe with new pole latitude and longitude columns.

#     Example usage:
#     ```python
#     df = polpts(df, 'pt1_lat', 'pt1_lon', 'pt2_lat', 'pt2_lon', 'pole_lat', 'pole_lon')
#     ```
#     """

#     dfn = df.copy()

#     # Convert first and second points from spherical to Cartesian
#     dfn = sphcar(dfn, pt1_lat, pt1_lon, 'x1', 'y1', 'z1')
#     dfn = sphcar(dfn, pt2_lat, pt2_lon, 'x2', 'y2', 'z2')

#     # Compute the normal vector to the great circle containing the two points
#     dot_product = dfn['x1'] * dfn['x2'] + dfn['y1'] * dfn['y2'] + dfn['z1'] * dfn['z2']
#     z = np.sin(np.degrees(np.arccos(dot_product)))

#     dfn['z1'] = (dfn['y1'] * dfn['z2'] - dfn['z1'] * dfn['y2']) / z
#     dfn['z2'] = (dfn['z1'] * dfn['x2'] - dfn['x1'] * dfn['z2']) / z
#     dfn['z3'] = (dfn['x1'] * dfn['y2'] - dfn['y1'] * dfn['x2']) / z

#     # Convert the computed Cartesian normal vector back to spherical coordinates
#     dfn = carsph(dfn, 'z1', 'z2', 'z3', pole_lat, pole_lon)

#     # Drop intermediate Cartesian coordinate columns
#     dfn = dfn.drop(columns=['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'z1', 'z2', 'z3'])

#     return dfn


############################################################################################################
def polpts(df, pt1_lat, pt1_lon, pt2_lat, pt2_lon, pole_lat, pole_lon):
    """
    Function polpts finds a pole given two points on a great circle.
    The input and output latitudes and longitudes are in **degrees**.

    The returned pole is the RIGHT-HANDED pole to the great circle from point 1 -> point 2,
    i.e., pole vector = (unit_vector(pt1) × unit_vector(pt2)) normalized.

    Args:
        df (DataFrame): Input dataframe containing the two points.
        pt1_lat (str): Column name for latitude of the first point.
        pt1_lon (str): Column name for longitude of the first point.
        pt2_lat (str): Column name for latitude of the second point.
        pt2_lon (str): Column name for longitude of the second point.
        pole_lat (str): Column name for output pole latitude.
        pole_lon (str): Column name for output pole longitude.

    Returns:
        DataFrame: A copy of the input dataframe with new pole latitude and longitude columns.

    Example usage:
        df = polpts(df, 'pt1_lat', 'pt1_lon', 'pt2_lat', 'pt2_lon', 'pole_lat', 'pole_lon')
    """
    dfn = df.copy()

    # Convert the two points from spherical to Cartesian (unit vectors)
    dfn = sphcar(dfn, pt1_lat, pt1_lon, '_x1_polpts', '_y1_polpts', '_z1_polpts')
    dfn = sphcar(dfn, pt2_lat, pt2_lon, '_x2_polpts', '_y2_polpts', '_z2_polpts')

    # Cross product (right-handed): p = a × b
    dfn['_px_polpts'] = dfn['_y1_polpts'] * dfn['_z2_polpts'] - dfn['_z1_polpts'] * dfn['_y2_polpts']
    dfn['_py_polpts'] = dfn['_z1_polpts'] * dfn['_x2_polpts'] - dfn['_x1_polpts'] * dfn['_z2_polpts']
    dfn['_pz_polpts'] = dfn['_x1_polpts'] * dfn['_y2_polpts'] - dfn['_y1_polpts'] * dfn['_x2_polpts']

    # Normalize the pole vector; handle degenerate cases (coincident or antipodal points)
    dfn['_pnorm_polpts'] = np.sqrt(dfn['_px_polpts']**2 + dfn['_py_polpts']**2 + dfn['_pz_polpts']**2)

    degenerate = dfn['_pnorm_polpts'] < 1e-15
    if degenerate.any():
        # No unique great circle pole if points are identical or exactly antipodal.
        # Leave outputs as NaN for those rows.
        dfn.loc[degenerate, pole_lat] = np.nan
        dfn.loc[degenerate, pole_lon] = np.nan

    ok = ~degenerate
    if ok.any():
        dfn.loc[ok, '_px_polpts'] = dfn.loc[ok, '_px_polpts'] / dfn.loc[ok, '_pnorm_polpts']
        dfn.loc[ok, '_py_polpts'] = dfn.loc[ok, '_py_polpts'] / dfn.loc[ok, '_pnorm_polpts']
        dfn.loc[ok, '_pz_polpts'] = dfn.loc[ok, '_pz_polpts'] / dfn.loc[ok, '_pnorm_polpts']

        # Convert pole vector back to spherical coordinates (degrees)
        dfn_ok = dfn.loc[ok].copy()
        dfn_ok = carsph(dfn_ok, '_px_polpts', '_py_polpts', '_pz_polpts', pole_lat, pole_lon)

        # Write back
        dfn.loc[ok, pole_lat] = dfn_ok[pole_lat]
        dfn.loc[ok, pole_lon] = dfn_ok[pole_lon] % 360.0  # keep consistent with your longitude usage

    # Drop intermediate columns
    dfn = dfn.drop(columns=[
        '_x1_polpts', '_y1_polpts', '_z1_polpts',
        '_x2_polpts', '_y2_polpts', '_z2_polpts',
        '_px_polpts', '_py_polpts', '_pz_polpts',
        '_pnorm_polpts'
    ])

    return dfn


############################################################################################################
def rot1(df, oldlat, oldlon, angle, newlat, newlon):
    """
    Rotate 'angle' about the x axis.

    Args:
        oldlat (float): Initial latitude in radians.
        oldlon (float): Initial longitude in radians.
        angle (float): Rotation angle in radians.

    Returns:
        newlat (float): New latitude after rotation.
        newlon (float): New longitude after rotation.
        
    All angles are in degrees.
    
    Example call
    df = rot1(df, 'Old_Lat', 'Old_Lon', 'Angle', 'New_Lat', 'New_Lon')
    """
    dfn = df.copy()
    dfn['ninety_deg'] = 90.0
    dfn['minus_ninety_deg'] = -90.0
     
    # First rotation
    dfn = rot2(dfn, oldlat, oldlon, 'minus_ninety_deg', 'tlat_rot1', 'tlon_rot1')
    # Second rotation
    dfn = rot3(dfn, 'tlat_rot1', 'tlon_rot1', angle, 'tlat2_rot1', 'tlon2_rot1')
    # Third rotation to get the new latitude and longitude
    dfn = rot2(dfn, 'tlat2_rot1', 'tlon2_rot1', 'ninety_deg', newlat, newlon)
    
    # Drop the intermediate columns
    dfn = dfn.drop(columns=['tlat_rot1', 'tlon_rot1', 'tlat2_rot1', 'tlon2_rot1', 'ninety_deg', 'minus_ninety_deg'])
    return dfn

############################################################################################################
def rot2(df, lat1, lon1, rotation, rot2_lat2, rot2_lon2):
    """
    Perform rotation about the y-axis in spherical coordinates.
    All angles are in degrees.
    
    Example call:
    df = rot2(df, 'oldlat', 'oldlon', 'angle', 'newlat', 'newlon')
    """
    dfn = df.copy()
    dfn = sphcar(dfn, lat1, lon1, 'x', 'y', 'z')
    dfn['x0'] = dfn['x'] 
    dfn['x'] = dfn['x0'] * np.cos(np.radians(dfn[rotation])) + dfn['z'] * np.sin(np.radians(dfn[rotation]))
    dfn['z'] = -dfn['x0'] * np.sin(np.radians(dfn[rotation])) + dfn['z'] * np.cos(np.radians(dfn[rotation]))
    dfn = carsph(dfn, 'x', 'y', 'z', rot2_lat2, rot2_lon2)
    
    # Drop the intermediate columns
    dfn = dfn.drop(columns=['x0','x', 'y', 'z'])
    return dfn

############################################################################################################
def rot3(df, lat1, lon1, rotation, rot3_lat2, rot3_lon2):
    """
    Perform rotation about the z-axis in spherical coordinates.
    All angles are in degrees.
    
    Example call:
    df = rot3(df, 'oldlat', 'oldlon', 'angle', 'newlat', 'newlon')
    """
    dfn = df.copy()
    dfn[rot3_lat2] = dfn[lat1]
    dfn[rot3_lon2] = (dfn[lon1] + dfn[rotation]) % 360.0
    return dfn

############################################################################################################
def rotate(df, lat_in, lon_in, rot_lat, rot_lon, rot_angle, lat_out, lon_out):
    """
    Rotate a point (lat, lon) around a rotation pole (rot_lat, rot_lon) by a given angle.
    The input and output latitudes, longitudes, and rotation angles are in **degrees**.

    Args:
        df (DataFrame): Input dataframe containing the point and rotation parameters.
        lat_in (str): Column name for latitude of the input point.
        lon_in (str): Column name for longitude of the input point.
        rot_lat (str): Column name for latitude of the rotation pole.
        rot_lon (str): Column name for longitude of the rotation pole.
        rot_angle (str): Column name for rotation angle in degrees (counterclockwise).
        lat_out (str): Column name for output latitude.
        lon_out (str): Column name for output longitude.

    Returns:
        DataFrame: A copy of the input dataframe with new rotated latitude and longitude columns.

    Example usage:
    ```python
    df = rotate(df, 'Lat_in', 'Lon_in', 'Rot_Lat', 'Rot_Lon', 'Rot_Angle', 'Lat_out', 'Lon_out')
    ```
    """

    dfn = df.copy()

    # Convert input coordinates and rotation pole to Cartesian
    dfn = sphcar(dfn, lat_in, lon_in, 'x', 'y', 'z')
    dfn = sphcar(dfn, rot_lat, rot_lon, 'rx', 'ry', 'rz')

    # Convert rotation angle to radians
    dfn['theta'] = np.radians(dfn[rot_angle])

    # Compute the rotation matrix using Rodrigues' rotation formula
    kx, ky, kz = dfn['rx'], dfn['ry'], dfn['rz']  # Rotation pole in Cartesian
    c = np.cos(dfn['theta'])
    s = np.sin(dfn['theta'])
    v = 1 - c

    dfn['r11'] = kx * kx * v + c
    dfn['r12'] = kx * ky * v - kz * s
    dfn['r13'] = kx * kz * v + ky * s
    dfn['r21'] = ky * kx * v + kz * s
    dfn['r22'] = ky * ky * v + c
    dfn['r23'] = ky * kz * v - kx * s
    dfn['r31'] = kz * kx * v - ky * s
    dfn['r32'] = kz * ky * v + kx * s
    dfn['r33'] = kz * kz * v + c

    # Apply the rotation matrix
    dfn['x_rot'] = dfn['r11'] * dfn['x'] + dfn['r12'] * dfn['y'] + dfn['r13'] * dfn['z']
    dfn['y_rot'] = dfn['r21'] * dfn['x'] + dfn['r22'] * dfn['y'] + dfn['r23'] * dfn['z']
    dfn['z_rot'] = dfn['r31'] * dfn['x'] + dfn['r32'] * dfn['y'] + dfn['r33'] * dfn['z']

    # Convert back to spherical coordinates
    dfn = car3sp(dfn, 'x_rot', 'y_rot', 'z_rot', lat_out, lon_out, 'radius')

    # Drop intermediate columns
    dfn = dfn.drop(columns=['x', 'y', 'z', 'rx', 'ry', 'rz', 'theta', 
                             'r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33',
                             'x_rot', 'y_rot', 'z_rot', 'radius'])

    return dfn

############################################################################################################
def rotaz(df, lat_in, lon_in, az_in, pole_lat, pole_lon, pole_ang, lat_out, lon_out, az_out):
    """
    Function rotaz finds the new location and azimuth after a rotation.
    The input and output latitudes, longitudes, and azimuths are in **degrees**.
    Azimuths are measured **clockwise from North**.

    Args:
        df (DataFrame): Input dataframe containing the original location and rotation parameters.
        lat_in (str): Column name for latitude before rotation.
        lon_in (str): Column name for longitude before rotation.
        az_in (str): Column name for azimuth before rotation.
        pole_lat (str): Column name for latitude of the rotation pole.
        pole_lon (str): Column name for longitude of the rotation pole.
        pole_ang (str): Column name for the rotation angle (degrees counterclockwise).
        lat_out (str): Column name for latitude after rotation (output).
        lon_out (str): Column name for longitude after rotation (output).
        az_out (str): Column name for azimuth after rotation (output).

    Returns:
        DataFrame: A copy of the input dataframe with new location and azimuth after rotation.
    """

    dfn = df.copy()
    
    dfn['adjusted_az'] = dfn[az_in] - 90.0
    dfn['ninety_deg'] = 90.0  # Define ninety_deg as a constant column
    
    # print("\nStep 1: Before locate()")
    # print(dfn[[lat_in, lon_in, 'adjusted_az']].head())  # Debugging print

    # Step 1: Compute location of old transform pole (90° from original point along az_in - 90°)
    dfn = locate(dfn, lat_in, lon_in, 'ninety_deg', 'adjusted_az', 'tlat1_rotaz', 'tlon1_rotaz')

    # print("\nStep 1: After locate()")
    # print(dfn.head())  # Debugging print

    # Step 2: Rotate location
    dfn = rotp(dfn, pole_lat, pole_lon, pole_ang, lat_in, lon_in, lat_out, lon_out)

    # print("\nStep 2: After rotp() on location")
    # print(dfn[[lat_out, lon_out]].head())  # Debugging print

    # Step 3: Rotate transform pole
    dfn = rotp(dfn, pole_lat, pole_lon, pole_ang, 'tlat1_rotaz', 'tlon1_rotaz', 'tlat2_rotaz', 'tlon2_rotaz')

    # print("\nStep 3: After rotp() on transform pole")
    # print(dfn[['tlat2_rotaz', 'tlon2_rotaz']].head())  # Debugging print

    # Step 4: Calculate new azimuth
    dfn = aztran(dfn, 'tlat2_rotaz', 'tlon2_rotaz', lat_out, lon_out, az_out)

    # print("\nStep 4: After aztran()")
    # print(dfn[[az_out]].head())  # Debugging print
    # print(dfn.head())

    # Drop intermediate columns
    dfn = dfn.drop(columns=['tlat1_rotaz', 'tlon1_rotaz', 'tlat2_rotaz', 'tlon2_rotaz', 'adjusted_az', 'ninety_deg'])

    return dfn


############################################################################################################
def rotp(df, pole_lat, pole_lon, rot_angle, old_lat, old_lon, new_lat, new_lon):
    """
    Function `rotp` rotates a point (old_lat, old_lon) by a given angle around an arbitrary Euler pole
    (pole_lat, pole_lon). The input and output latitudes, longitudes, and rotation angles are in degrees.

    Args:
        df (DataFrame): Input dataframe containing the point.
        pole_lat (str): Latitude of the Euler pole (degrees).
        pole_lon (str): Longitude of the Euler pole (degrees).
        rot_angle (str): Rotation angle in degrees (counterclockwise).
        old_lat (str): Latitude of the point to rotate.
        old_lon (str): Longitude of the point to rotate.
        new_lat (str): Latitude after rotation.
        new_lon (str): Longitude after rotation.

    Returns:
        DataFrame: A copy of the input dataframe with new rotated latitude and longitude columns.

    Example usage:
    ```python
    df = rotp(df, 'Pole_Lat', 'Pole_Lon', 'Angle', 'Old_Lat', 'Old_Lon', 'New_Lat', 'New_Lon')
    ```
    """

    dfn = df.copy()

    # Step 1: Rotate the point's longitude by -pole_lon
    dfn['neg_pole_lon'] = -dfn[pole_lon]
    dfn = rot3(dfn, old_lat, old_lon, 'neg_pole_lon', 'c1', 'd1')

    # Step 2: Rotate around the y-axis by (pole_lat - 90 degrees)
    dfn['pole_lat_minus_ninety'] = dfn[pole_lat] - 90.0
    dfn = rot2(dfn, 'c1', 'd1', 'pole_lat_minus_ninety', 'c2', 'd2')

    # Step 3: Rotate around the z-axis by the given rotation angle
    dfn = rot3(dfn, 'c2', 'd2', rot_angle, 'c3', 'd3')

    # Step 4: Rotate back around the y-axis by (90 degrees - pole_lat)
    dfn['ninety_minus_pole_lat'] = 90.0 - dfn[pole_lat]
    dfn = rot2(dfn, 'c3', 'd3', 'ninety_minus_pole_lat', 'c4', 'd4')

    # Step 5: Rotate back around the z-axis by +pole_lon to return to the original reference frame
    dfn = rot3(dfn, 'c4', 'd4', pole_lon, new_lat, new_lon)

    # Drop intermediate transformation columns
    dfn = dfn.drop(columns=['c1', 'd1', 'c2', 'd2', 'c3', 'd3', 'c4', 'd4', 'neg_pole_lon', 
                            'pole_lat_minus_ninety', 'ninety_minus_pole_lat'])

    return dfn

############################################################################################################
def smcir(plat, plon, angular_distance, num_points):
    """
    Generates a DataFrame containing latitude and longitude points forming a small circle
    around a given central latitude and longitude.
    
    Parameters:
    plat (float): Latitude of the center point in degrees
    plon (float): Longitude of the center point in degrees
    angular_distance (float): Angular radius of the small circle in degrees
    num_points (int): Number of points to generate for the circle (default: 361)
    
    Returns:
    pandas.DataFrame: DataFrame containing latitude and longitude columns of the small circle
    """
    
    # Generate circle points
    azimuths = np.linspace(180, -180, num_points) # Reverse the azimuths so the points are generated in a clockwise direction
    # print(f"The azimuths: {azimuths}")
    # Create a DataFrame to store the values
    df = pd.DataFrame(columns=['plat','plon','AngDis','Azimuth'])
    df['Azimuth'] = azimuths
    df['plat'] = plat
    df['plon'] = plon
    df['AngDis'] = angular_distance
    df = locate(df, 'plat', 'plon', 'AngDis', 'Azimuth', 'smcir_lat', 'smcir_lon')
    df = adjust_az(df, 'smcir_lon',0.0, 360.)
    # Drop the intermediate columns
    df = df.drop(columns=['plat', 'plon', 'AngDis', 'Azimuth'])
    return df

# Example Usage
# lat, lon = 30, 60  # Center point latitude and longitude
# angular_distance = 10  # Angular distance in degrees
# num_points = 361  # Number of points to generate
# df_circle = smcir(lat, lon, angular_distance, num_points)

############################################################################################################
def sphcar(df, inc_lat, dec_lon, x, y, z):
    """
    Function sphcar converts spherical coordinates to cartesian coordinates for a dataframe.
    The cartesian coordinates are added to the dataframe as columns 'x', 'y', and 'z'.
    The length of the vector is assumed to be 1. 
        If you know the length of the vector, you can multiply the output columns (x,y,z) by it.
    The spherical coordinates are assumed to be in degrees.
    The cartesian coordinates are in the same units as the vector.
    See https://en.wikipedia.org/wiki/Spherical_coordinate_system
    See https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_spherical_coordinates
    
    Example call
    # df = sphcar(df, 'Inclination', 'Declination','x', 'y', 'z')
    """

    dfn = df.copy()
    # Compute the Cartesian coordinates for each inclination, declination or latitude, longitude pair.
    dfn[x] = np.cos(np.radians(dfn[inc_lat])) * np.cos(np.radians(dfn[dec_lon]))
    dfn[y] = np.cos(np.radians(dfn[inc_lat])) * np.sin(np.radians(dfn[dec_lon]))
    dfn[z] = np.sin(np.radians(dfn[inc_lat]))
    dfn[z] = dfn[z].clip(lower=-1,upper=1)  # This keeps df['z'] between -1 and 1
    return dfn

############################################################################################################
def vector_mean(df, inc_lat, dec_lon):
    """
    Function vector_mean computes the vector mean for a dataframe with a set of inc, dec or lat, lon pairs
    and returns the mean inclination, declination and the vector mean length R.
    The input inclination, declination or latitude, longitude are in degrees.
    
    Example call
    vgp_lat_mean, vgp_lon_mean, R = vector_mean(df, 'VGP_Lat', 'VGP_Lon')
    """
    # Column names in call to function must be in the dataframe
    # for col in [inc_lat, dec_lon]:
    #     if col not in df.columns:
    #         print(f"Column '{col}' not found in the DataFrame.")
    #         return np.nan, np.nan, np.nan
    for col in [inc_lat, dec_lon]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")
    
    dfn = df.copy()
    # Convert the spherical coordinates to cartesian coordinates.
    dfn = sphcar(dfn, inc_lat, dec_lon,'x', 'y', 'z')
    # Compute the vector sum and mean. Note, the mean is not really needed.
    x_sum = dfn['x'].sum()
    y_sum = dfn['y'].sum()
    z_sum = dfn['z'].sum()
    # x_mean = dfn['x'].mean()
    # y_mean = dfn['y'].mean()
    # z_mean = dfn['z'].mean()
    
    # Compute the vector mean length.
    # R = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2) * len(dfn) 
    # R above also = np.sqrt(x_sum**2 + y_sum**2 + z_sum**2)
    # print(f"Vector mean length: {R}")
    R = np.sqrt(x_sum**2 + y_sum**2 + z_sum**2)
    # print(f"Vector mean length: {R}")
    # print(f"N: {len(dfn)}")
    if R == 0.0:
        return np.nan, np.nan, 0.0
    x_norm = x_sum / R
    y_norm = y_sum / R
    z_norm = z_sum / R
    
    # Compute the vector mean
    inc_lat_mean = np.degrees(np.arcsin(z_norm))
    dec_lon_mean = np.degrees(np.arctan2(y_norm, x_norm))
    
    # Drop the intermediate columns
    dfn = dfn.drop(columns=['x', 'y', 'z'])
    
    # Ensure the declination is in the range [0, 360)
    dec_lon_mean = dec_lon_mean % 360
    
    # Return the vector mean inclination and declination and the vector mean length R.
    return inc_lat_mean, dec_lon_mean, R


############################################################################################################
def vgp(df, slat, slon, inc, dec, colat, plat, plon):
    """
    Calculate Virtual Geomagnetic Poles (VGPs) from inclination and declination.
    All angles are in degrees.
    
    Example call
    df = vgp(df, 'Site_Lat', 'Site_Lon', 'Inc_All', 'Dec_All', 'VGP_Lat', 'VGP_Lon', 'Colat')
    """
    dfn = df.copy()
    # Calculate colatitude
    dfn[colat] = np.degrees(np.arctan(2 / np.tan(np.radians(dfn[inc]))))
    
    # If colatitude is <0°, make it colatitude + 180°
    dfn.loc[dfn[colat] < 0, colat] = dfn[colat] + 180
    
    # Calculate VGP latitude and longitude
    dfn = locate(dfn, slat, slon, colat, dec, plat, plon)
    
    # If colatitude is >180° or <0°, write a warning message
    if dfn[colat].max() > 180 or dfn[colat].min() < 0:
        print(f"Warning: Colatitude is >180° or <0°.")
    return dfn
############################################################################################

