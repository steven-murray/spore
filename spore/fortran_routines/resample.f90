subroutine lay_ps_map(Nnu,Nbl,Ngrid, nu, uvsample, baselines, ugrid, vgrid, psmap, radmin )
    implicit none
    
    integer, intent(in) :: Nnu, Nbl, Ngrid
    real(8), intent(in) :: nu(Nnu)
    real(8), intent(in) :: uvsample(Nbl, Nnu)
    real(8), intent(in) :: ugrid(Ngrid,Ngrid), vgrid(Ngrid,Ngrid)
    real(8), intent(in) :: baselines(Nbl,2)
    real(8), intent(in) :: radmin
            
    real(8), intent(out) :: psmap(Ngrid,Ngrid, Nnu)
            
    integer :: i,j, ncells, tot_ncells, n_u,n_v, ui0,uif,vi0,vif
    real(8) :: uv(Nbl,2)
    real(8) :: sg(Nnu)
    real(8) :: pi = 3.141592653589
    real(8) :: du,dv, umin,vmin, rmin, umax,vmax
    integer :: u_ind_min, u_ind_max, v_ind_min,v_ind_max
    !real(8) :: beammap(Ngrid,Ngrid)
    real(8),allocatable :: u2(:,:), beam(:,:) ,beammap(:,:)
    

    psmap = 0.d0

    du = ugrid(1,2)-ugrid(1,1)
    dv = vgrid(2,1)-vgrid(1,1)
    umin = minval(ugrid)
    vmin = minval(vgrid)
    umax = maxval(ugrid)
    vmax = maxval(vgrid)
    
    sg = 0.42*3e8/(nu*1e6*4.0)
    rmin = radmin*maxval(1./(4*pi**2*sg**2))
    ncells = int(ceiling(rmin/du))
    tot_ncells = 2*ncells+1

        
    !$OMP PARALLEL PRIVATE(i,j,n_u,n_v,ui0,uif,vi0,vif,uv,u_ind_min,&
    !$OMP&           u_ind_max,v_ind_min,v_ind_max,beammap,u2,beam)&
    !$OMP& NUM_THREADS(8)

    allocate(beammap(Ngrid,Ngrid))
    beammap = 0.d0
    
    
    ! Initial allocation of u2, beam
    allocate(u2(tot_ncells,tot_ncells), beam(tot_ncells, tot_ncells))
    
    !$OMP DO 
    do i=1,Nnu
        uv = baselines * nu(i)/3e2
        do j = 1,Nbl
            if ((uv(j,1) + rmin < umin).or.(uv(j,2)+ rmin<vmin) .or. &
                (uv(j,1) - rmin > umax).or.(uv(j,2)- rmin>vmax)) then
                cycle
            end if
            
            u_ind_min = nint((uv(j,1)- umin)/du) + 1 - ncells
            u_ind_max = u_ind_min + tot_ncells
            v_ind_min = nint((uv(j,2)- vmin)/dv) + 1 - ncells
            v_ind_max = v_ind_min + tot_ncells
            
            ui0 = max(u_ind_min,1)
            uif = min(u_ind_max,Ngrid)
            vi0 = max(v_ind_min,1)
            vif = min(v_ind_max,Ngrid)
            
            n_u = uif - ui0 + 1
            n_v = vif - vi0 + 1

            if (size(u2,1)/=n_u .or. size(u2,2)/=n_v) then
                deallocate(u2,beam)
                allocate(u2(n_u,n_v),beam(n_u,n_v))
            end if
            
            !if (n_u<tot_ncells .or. n_v<tot_ncells) then
            !    write(*,*) n_u,n_v, vi0,vif,ui0,uif
            !end if
            
            u2 = (uv(j,1) - ugrid(vi0:vif,ui0:uif))**2 +&
                 (uv(j,2) - vgrid(vi0:vif,ui0:uif))**2
            
            beam = exp(-2*pi**2 * sg(i)**2 * u2)
            psmap(vi0:vif,ui0:uif,i) = psmap(vi0:vif,ui0:uif,i) + uvsample(j,i)*beam
            beammap(vi0:vif,ui0:uif) = beammap(vi0:vif,ui0:uif)+beam
        end do
        psmap(:,:,i) = psmap(:,:,i)/beammap
    end do
    !$OMP END DO
    !$OMP END PARALLEL
end subroutine
