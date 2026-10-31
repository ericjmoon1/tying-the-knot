"""Check the construction endpoints, section ordering and retained link.

Run: python code/verify_geometry.py
Prints check results without creating additional files.
The sampling checks are numerical evidence, not a continuum proof.
"""
import numpy as np
import geometry as g

if __name__=='__main__':
    theta=np.linspace(0,2*np.pi,257)
    beta=np.linspace(0,2*np.pi,1025)
    tt,bb=np.meshgrid(theta,beta,indexing='ij')
    v,u=g.perimeter(bb)
    assert np.allclose(g.construction_points(tt,bb,0),g.thick_ribbon(tt,u,v),atol=1e-13)
    assert np.array_equal(g.construction_points(tt,bb,1),g.refined_points(tt,bb))
    x,z=g.refined_section_coordinates(tt,bb)
    a1=g.lifted_angle(x,z,bb+2*tt)
    a0=g.lifted_angle(v,u,bb)+2*tt
    r0=np.hypot(v,u);r1=np.hypot(x,z)
    for progress in np.linspace(0,1,11):
        angle=(1-progress)*a0+progress*a1
        radius=(1-progress)*r0+progress*r1
        assert np.min(np.diff(angle,axis=1))>0
        assert np.min(radius)>0 and np.max(radius)<g.RADIUS
        surface=g.construction_points(tt,bb,float(progress))
        assert np.allclose(surface[0],surface[-1],atol=1e-12)
        assert np.allclose(surface[:,0],surface[:,-1],atol=1e-12)
    for label in [np.pi/2,3*np.pi/2]:
        b=np.full_like(theta,label)
        assert np.array_equal(g.construction_points(theta,b,1),g.ring_points(theta,b))
        assert np.allclose(g.face_adjusted_points(theta,b,.030,.065),g.ring_points(theta,b),atol=1e-14)
    S=np.stack([(g.RADIUS-u*np.sin(2*tt))*np.cos(tt),
                (g.RADIUS-u*np.sin(2*tt))*np.sin(tt),u*np.cos(2*tt)],axis=-1)
    assert np.allclose(g.thick_ribbon(tt,u,np.zeros_like(u)),S)
    print('PASS: thick-ribbon boundary at progress=0; S is the v=0 slice.')
    print('PASS: adjusted surface at progress=1; exact original K1,K2 retained.')
    print('PASS: sampled sections retain angular order, positive radius and closure.')
    print('PASS: boundary-vanishing face adjustment retains K1,K2 to roundoff.')
