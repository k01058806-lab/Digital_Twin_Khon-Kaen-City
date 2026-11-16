import numpy as np, glob, json, math, time, os
from scipy import ndimage
import contourpy, cv2
EPS=0.35*0.00018064
def simp(r):
    o=np.array([102.7,16.3]);q=cv2.approxPolyDP(((r-o)/EPS*4).astype(np.float32).reshape(-1,1,2),4,True).reshape(-1,2).astype(np.float64)/4*EPS+o
    return q
import sys; src=sys.argv[1]
files=sorted(glob.glob(src+'/*.asc'), key=lambda s:s.split('_')[-1])
x0,y0,cs,nc,nr=102.7358703613,16.3479690552,0.00018064,1037,1000
N=y0+nr*cs
BANDS=[0.05,0.15,0.3,0.5,1.0,2.0,99]
os.makedirs('hires',exist_ok=True)
for f in files:
    n=f.split('_')[-1][:5]; t=time.time()
    g=np.loadtxt(f,skiprows=6); g[g==-9999]=0
    # 2x cubic upsample + light smoothing for contour generation
    z=ndimage.zoom(g,2,order=3); z=ndimage.gaussian_filter(z,0.6); z[z<0]=0
    zc=cs/2
    xs=x0+(np.arange(z.shape[1])+0.5)*zc; ys=N-(np.arange(z.shape[0])+0.5)*zc
    gen=contourpy.contour_generator(xs,ys,z,fill_type=contourpy.FillType.OuterOffset,line_type=None)
    feats=[]; npts=0
    for lo,hi in zip(BANDS[:-1],BANDS[1:]):
        polys,offs=gen.filled(lo,hi)
        for pts,off in zip(polys,offs):
            rings=[pts[off[i]:off[i+1]] for i in range(len(off)-1)]
            outer=rings[0]
            # area in cells (shoelace)
            a=0.5*abs(np.dot(outer[:,0],np.roll(outer[:,1],1))-np.dot(outer[:,1],np.roll(outer[:,0],1)))/(cs*cs)
            if a<3: continue
            rr=[np.vstack([q,q[:1]]) for q in (simp(r) for r in rings)]
            rr=[np.round(r,5).tolist() for r in rr if len(r)>=4]
            if not rr or len(rings[0])<4: continue
            npts+=sum(len(r) for r in rr)
            feats.append({'type':'Feature','properties':{'d':lo,'h':hi if hi<99 else 3.3},'geometry':{'type':'Polygon','coordinates':rr}})
    js=json.dumps({'type':'FeatureCollection','features':feats},separators=(',',':'))
    open(f'hires/flood_{n}.geojson','w').write(js)
    print(n,len(feats),'polys',npts,'pts',round(len(js)/1e6,1),'MB',round(time.time()-t,1),'s')
