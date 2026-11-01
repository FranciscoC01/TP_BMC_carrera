"""
lectura_datos.py - Script base para el Práctico de Biomecánica
================================================================

Lee los archivos de captura de movimiento del práctico y hace un gráfico
rápido para verificar que se cargaron bien. NO resuelve ninguna consigna:
filtrado, centros de masa, ángulos, etc. quedan a cargo de cada grupo.

Requisitos: Python 3.8+, numpy, pandas, matplotlib (sin otras dependencias).

Uso desde la terminal:
    python lectura_datos.py "Salto CMJ/Salto CMJ.trc"
    python lectura_datos.py "Salto CMJ/Salto CMJ.mot"
    python lectura_datos.py "Salto CMJ/Salto CMJ.c3d"

Uso como módulo (en otro script o en un notebook):
    from lectura_datos import leer_trc, leer_mot, leer_c3d, marcador_2d
    trc = leer_trc("Salto CMJ/Salto CMJ.trc")
    x, y = marcador_2d(trc, "r knee")        # plano sagital (X avance, Y vertical), en mm
    grf = leer_mot("Salto CMJ/Salto CMJ.mot")  # fuerzas en N, a la frecuencia del TRC
    c3d = leer_c3d("Salto CMJ/Salto CMJ.c3d")  # fuerzas a la frecuencia original (p. ej. 1000 Hz)

Sistema de referencia del laboratorio:
    Y = vertical (hacia arriba) en todos los archivos.
    En piso: X = dirección de avance, Z = medio-lateral.
    Excepciones (cinta): GaitTM avanza hacia -Z (plano sagital = Z-Y) y
    CARRERA avanza hacia -X. Ver el README.
    Marcadores en mm. Fuerzas en N. Tiempo en s.

Diferencias entre formatos:
    .trc  -> marcadores ya recortados y SIN huecos (rellenados por el software).
    .c3d  -> marcadores crudos, CON huecos (NaN): usarlos para la consigna de interpolación.
    .mot  -> fuerzas en el sistema del laboratorio (vy = vertical), remuestreadas a 100 Hz.
    .c3d  -> fuerzas a la frecuencia original, en el sistema de CADA plataforma:
             vertical = -Fz; momentos en N·mm.
"""

import struct
import sys

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# TRC: trayectorias de marcadores (texto, formato OpenSim)
# ----------------------------------------------------------------------------
def leer_trc(ruta):
    """Lee un .trc y devuelve un dict con:
    'frec' (Hz), 'tiempo' (array), 'marcadores' (lista de nombres),
    'datos' (DataFrame con columnas '<marcador>_X', '_Y', '_Z' en mm).
    Las celdas vacías (huecos) quedan como NaN.
    """
    with open(ruta, encoding="latin1") as f:
        lineas = f.read().splitlines()
    frec = float(lineas[2].split("\t")[0])
    nombres = [n.strip() for n in lineas[3].split("\t")[2:] if n.strip()]
    ncol = 2 + 3 * len(nombres)
    filas = []
    for l in lineas[5:]:
        if not l.strip():
            continue
        partes = l.split("\t")[:ncol]
        partes += [""] * (ncol - len(partes))
        filas.append([float(p) if p.strip() else np.nan for p in partes])
    A = np.array(filas)
    columnas = [f"{m}_{eje}" for m in nombres for eje in "XYZ"]
    datos = pd.DataFrame(A[:, 2:], columns=columnas)
    return {"frec": frec, "tiempo": A[:, 1], "marcadores": nombres, "datos": datos}


def marcador_2d(trc, nombre):
    """Devuelve (x, y) del marcador en el plano sagital: X avance, Y vertical (mm)."""
    d = trc["datos"]
    return d[f"{nombre}_X"].to_numpy(), d[f"{nombre}_Y"].to_numpy()


# ----------------------------------------------------------------------------
# MOT: fuerzas de reacción del piso (texto, formato OpenSim exportado por BTS)
# ----------------------------------------------------------------------------
def leer_mot(ruta):
    """Lee un .mot de fuerzas y devuelve un DataFrame.
    Columnas: time, r_gr_force_vx/vy/vz (N), r_gr_force_px/py/pz (centro de
    presión, m), r_gr_torque_x/y/z (N·m) y lo mismo para 'l_' (pie izquierdo).
    vy es la fuerza vertical. Atención: la frecuencia del .mot es la del .trc
    (100 Hz en la mayoría); para RFD, impulso o despegue preciso usar el .c3d.
    """
    with open(ruta, encoding="latin1") as f:
        lineas = f.read().splitlines()
    k = next(i for i, l in enumerate(lineas) if l.strip().lower() == "endheader")
    columnas = lineas[k + 1].split("\t")
    datos = np.loadtxt(lineas[k + 2:])
    return pd.DataFrame(datos, columns=columnas)


# ----------------------------------------------------------------------------
# C3D: formato binario estándar (marcadores + canales analógicos)
# ----------------------------------------------------------------------------
def _float_dec(raw):
    """Convierte floats en formato DEC (VAX) a IEEE."""
    a = np.frombuffer(raw, dtype="<u2").reshape(-1, 2)[:, ::-1].copy()
    v = np.frombuffer(a.tobytes(), dtype=">u4").byteswap().view("<f4") / 4.0
    return v


def _parametros_c3d(b):
    bloque = b[0]
    ini = (bloque - 1) * 512
    proc = b[ini + 3]  # 84 = Intel, 85 = DEC, 86 = MIPS
    e = ">" if proc == 86 else "<"
    grupos, params = {}, {}
    i = ini + 4
    while True:
        nlen = struct.unpack("b", b[i:i + 1])[0]
        gid = struct.unpack("b", b[i + 1:i + 2])[0]
        if nlen == 0:
            break
        nombre = b[i + 2:i + 2 + abs(nlen)].decode("latin1").upper()
        j = i + 2 + abs(nlen)
        salto = struct.unpack(e + "h", b[j:j + 2])[0]
        if gid < 0:
            grupos[-gid] = nombre
        else:
            k = j + 2
            tipo = struct.unpack("b", b[k:k + 1])[0]
            nd = b[k + 1]
            dims = list(b[k + 2:k + 2 + nd])
            k += 2 + nd
            n = int(np.prod(dims)) if dims else 1
            raw = b[k:k + n * abs(tipo)]
            if tipo == -1:
                if nd <= 1:
                    val = raw.decode("latin1").strip()
                else:
                    w = dims[0]
                    val = [raw[q * w:(q + 1) * w].decode("latin1").strip()
                           for q in range(n // w)] if w else []
            elif tipo == 1:
                val = list(raw)
            elif tipo == 2:
                val = list(struct.unpack(e + "%dh" % n, raw))
            else:
                val = list(_float_dec(raw)) if proc == 85 else list(struct.unpack(e + "%df" % n, raw))
            params[(gid, nombre)] = val
        if salto == 0:
            break
        i = j + salto
    out = {}
    for (g, n), v in params.items():
        out[grupos.get(g, str(g)) + ":" + n] = v
    return out, proc


def leer_c3d(ruta):
    """Lee un .c3d y devuelve un dict con:
    'frec_marcadores', 'marcadores' (nombres), 'puntos' (array frames x marcadores x 3, mm,
    NaN donde el marcador no fue visto), 'frec_analog', 'canales' (nombres),
    'analog' (DataFrame con los canales analógicos a su frecuencia original),
    'parametros' (todos los parámetros del archivo, por si hacen falta).
    """
    with open(ruta, "rb") as f:
        b = f.read()
    P, proc = _parametros_c3d(b)
    e = ">" if proc == 86 else "<"
    h = struct.unpack(e + "10h", b[0:20])
    primer, ultimo = h[3], h[4]
    if ultimo < primer:  # archivos largos: el header desborda int16
        ultimo += 65536
    nframes = ultimo - primer + 1

    npts = P["POINT:USED"][0]
    escala = P["POINT:SCALE"][0]
    frec = P["POINT:RATE"][0]
    nana = P.get("ANALOG:USED", [0])[0]
    frec_an = P.get("ANALOG:RATE", [0.0])[0] or 0.0
    ratio = int(round(frec_an / frec)) if nana else 0
    inicio = (P["POINT:DATA_START"][0] - 1) * 512

    es_float = escala < 0
    tam = 4 if es_float else 2
    por_frame = npts * 4 + nana * ratio
    raw = b[inicio:inicio + nframes * por_frame * tam]
    nframes = len(raw) // (por_frame * tam)
    raw = raw[:nframes * por_frame * tam]
    if es_float:
        v = _float_dec(raw) if proc == 85 else np.frombuffer(raw, dtype=e + "f4")
    else:
        v = np.frombuffer(raw, dtype=e + "i2").astype(float)
    v = v.reshape(nframes, por_frame)

    pts = v[:, :npts * 4].reshape(nframes, npts, 4).astype(float)
    xyz = pts[:, :, :3] * (1.0 if es_float else abs(escala))
    residuo = pts[:, :, 3]
    xyz[residuo < 0] = np.nan  # marcador no visto
    # Marcadores sin dato exportados como (0,0,0): también son huecos
    xyz[np.all(pts[:, :, :3] == 0, axis=2)] = np.nan

    nombres = (P.get("POINT:LABELS", []) + P.get("POINT:LABELS2", []))[:npts]
    salida = {"frec_marcadores": frec, "marcadores": nombres, "puntos": xyz,
              "frec_analog": frec_an, "canales": [], "analog": None, "parametros": P}

    if nana:
        an = v[:, npts * 4:].reshape(nframes * ratio, nana).astype(float)
        off = np.array(P.get("ANALOG:OFFSET", [0] * nana)[:nana], dtype=float)
        if P.get("ANALOG:FORMAT", "SIGNED").upper().startswith("UNSIGNED") and not es_float:
            an = np.where(an < 0, an + 65536, an)
        esc = np.array(P.get("ANALOG:SCALE", [1.0] * nana)[:nana], dtype=float)
        gen = P.get("ANALOG:GEN_SCALE", [1.0])[0]
        an = (an - off) * esc * gen
        canales = (P.get("ANALOG:LABELS", []) + P.get("ANALOG:LABELS2", []))[:nana]
        canales = [c if c else f"canal{i+1}" for i, c in enumerate(canales)]
        # nombres repetidos -> se agrega sufijo
        vistos = {}
        unicos = []
        for c in canales:
            vistos[c] = vistos.get(c, 0) + 1
            unicos.append(c if vistos[c] == 1 else f"{c}_{vistos[c]}")
        df = pd.DataFrame(an, columns=unicos)
        df.insert(0, "time", np.arange(len(df)) / frec_an)
        salida["canales"] = unicos
        salida["analog"] = df
    return salida


# ----------------------------------------------------------------------------
# Vista rápida
# ----------------------------------------------------------------------------
def _vista_rapida(ruta):
    import matplotlib.pyplot as plt

    ext = ruta.lower().rsplit(".", 1)[-1]
    if ext == "trc":
        trc = leer_trc(ruta)
        huecos = trc["datos"].isna().sum()
        huecos = huecos[huecos > 0]
        print(f"{len(trc['tiempo'])} frames a {trc['frec']:.0f} Hz, {len(trc['marcadores'])} marcadores:")
        print("  " + ", ".join(trc["marcadores"]))
        print("Huecos (NaN):", "ninguno" if huecos.empty else dict(huecos))
        fig, ax = plt.subplots(figsize=(10, 5))
        for m in trc["marcadores"]:
            ax.plot(trc["tiempo"], trc["datos"][f"{m}_Y"], label=m, lw=1)
        ax.set_xlabel("tiempo (s)")
        ax.set_ylabel("posición vertical Y (mm)")
        ax.set_title(ruta)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    elif ext == "mot":
        grf = leer_mot(ruta)
        dt = np.median(np.diff(grf["time"]))
        print(f"{len(grf)} muestras a {1/dt:.0f} Hz. Columnas: {list(grf.columns)}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(grf["time"], grf["r_gr_force_vy"], label="derecho Fy (vertical)")
        ax.plot(grf["time"], grf["l_gr_force_vy"], label="izquierdo Fy (vertical)")
        ax.plot(grf["time"], grf["r_gr_force_vx"], "--", lw=1, label="derecho Fx (antero-posterior)")
        ax.plot(grf["time"], grf["l_gr_force_vx"], "--", lw=1, label="izquierdo Fx (antero-posterior)")
        ax.set_xlabel("tiempo (s)")
        ax.set_ylabel("fuerza (N)")
        ax.set_title(ruta)
        ax.legend()
    elif ext == "c3d":
        c = leer_c3d(ruta)
        print(f"Marcadores: {len(c['marcadores'])} a {c['frec_marcadores']:.0f} Hz, "
              f"{c['puntos'].shape[0]} frames")
        print("  " + ", ".join(c["marcadores"]))
        if c["analog"] is None:
            print("Sin canales analógicos (no hay fuerzas en este archivo).")
            return
        print(f"Analógicos: {len(c['canales'])} canales a {c['frec_analog']:.0f} Hz")
        print("  " + ", ".join(c["canales"]))
        # En el .c3d las fuerzas están en el sistema de cada plataforma:
        # la vertical es Fz y es negativa (fuerza que el pie aplica sobre la plataforma).
        fz = [ch for ch in c["canales"] if ch.upper().startswith("FZ")]
        fig, ax = plt.subplots(figsize=(10, 5))
        for ch in fz:
            if np.abs(c["analog"][ch]).max() > 300:  # solo plataformas con señal
                ax.plot(c["analog"]["time"], -c["analog"][ch], label="-" + ch, lw=1)
        ax.set_xlabel("tiempo (s)")
        ax.set_ylabel("fuerza vertical (N)")
        ax.set_title(ruta + "  (vertical = -Fz de cada plataforma)")
        ax.legend()
    else:
        print("Formato no soportado. Usar .trc, .mot o .c3d")
        return
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        _vista_rapida(sys.argv[1])
