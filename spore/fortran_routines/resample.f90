!subroutine lay_ps_map(Nnu,Nbl,Ngrid, nu, uvsample, baselines, ugrid, vgrid, psmap, radmin )
!    implicit none
!
!    integer, intent(in) :: Nnu, Nbl, Ngrid
!    real(8), intent(in) :: nu(Nnu)
!    real(8), intent(in) :: uvsample(Nbl, Nnu)
!    real(8), intent(in) :: ugrid(Ngrid,Ngrid), vgrid(Ngrid,Ngrid)
!    real(8), intent(in) :: baselines(Nbl,2)
!    real(8), intent(in) :: radmin
!
!    real(8), intent(out) :: psmap(Ngrid,Ngrid, Nnu)
!
!    integer :: i,j, ncells, tot_ncells, n_u,n_v, ui0,uif,vi0,vif
!    real(8) :: uv(Nbl,2)
!    real(8) :: sg(Nnu)
!    real(8) :: pi = 3.141592653589
!    real(8) :: du,dv, umin,vmin, rmin, umax,vmax
!    integer :: u_ind_min, u_ind_max, v_ind_min,v_ind_max
!    !real(8) :: beammap(Ngrid,Ngrid)
!    real(8),allocatable :: u2(:,:), beam(:,:) ,beammap(:,:)
!
!
!    psmap = 0.d0
!
!    du = ugrid(1,2)-ugrid(1,1)
!    dv = vgrid(2,1)-vgrid(1,1)
!    umin = minval(ugrid)
!    vmin = minval(vgrid)
!    umax = maxval(ugrid)
!    vmax = maxval(vgrid)
!
!    sg = 0.42*3e8/(nu*1e6*4.0)
!    rmin = radmin*maxval(1./(4*pi**2*sg**2))
!    ncells = int(ceiling(rmin/du))
!    tot_ncells = 2*ncells+1
!
!
!    !$OMP PARALLEL PRIVATE(i,j,n_u,n_v,ui0,uif,vi0,vif,uv,u_ind_min,&
!    !$OMP&           u_ind_max,v_ind_min,v_ind_max,beammap,u2,beam)&
!    !$OMP& NUM_THREADS(8)
!
!    allocate(beammap(Ngrid,Ngrid))
!    beammap = 0.d0
!
!
!    ! Initial allocation of u2, beam
!    allocate(u2(tot_ncells,tot_ncells), beam(tot_ncells, tot_ncells))
!
!    !$OMP DO
!    do i=1,Nnu
!        uv = baselines * nu(i)/3e2
!        do j = 1,Nbl
!            if ((uv(j,1) + rmin < umin).or.(uv(j,2)+ rmin<vmin) .or. &
!                (uv(j,1) - rmin > umax).or.(uv(j,2)- rmin>vmax)) then
!                cycle
!            end if
!
!            u_ind_min = nint((uv(j,1)- umin)/du) + 1 - ncells
!            u_ind_max = u_ind_min + tot_ncells
!            v_ind_min = nint((uv(j,2)- vmin)/dv) + 1 - ncells
!            v_ind_max = v_ind_min + tot_ncells
!
!            ui0 = max(u_ind_min,1)
!            uif = min(u_ind_max,Ngrid)
!            vi0 = max(v_ind_min,1)
!            vif = min(v_ind_max,Ngrid)
!
!            n_u = uif - ui0 + 1
!            n_v = vif - vi0 + 1
!
!            if (size(u2,1)/=n_u .or. size(u2,2)/=n_v) then
!                deallocate(u2,beam)
!                allocate(u2(n_u,n_v),beam(n_u,n_v))
!            end if
!
!            !if (n_u<tot_ncells .or. n_v<tot_ncells) then
!            !    write(*,*) n_u,n_v, vi0,vif,ui0,uif
!            !end if
!
!            u2 = (uv(j,1) - ugrid(vi0:vif,ui0:uif))**2 +&
!                 (uv(j,2) - vgrid(vi0:vif,ui0:uif))**2
!
!            beam = exp(-2*pi**2 * sg(i)**2 * u2)
!            psmap(vi0:vif,ui0:uif,i) = psmap(vi0:vif,ui0:uif,i) + uvsample(j,i)*beam
!            beammap(vi0:vif,ui0:uif) = beammap(vi0:vif,ui0:uif)+beam
!        end do
!        psmap(:,:,i) = psmap(:,:,i)/beammap
!    end do
!    !$OMP END DO
!    !$OMP END PARALLEL
!end subroutine


subroutine grid_visibilities_gauss(Nnu,Nbl,Ngrid_u, Ngrid_v, f, visibilities, baselines, ugrid, vgrid, gridded_vis,&
                                   weights, radmin,static_beam, sigma0, nthreads )
    implicit none
    ! ==================================================================================================================
    ! grid visibilities
    !
    ! Take a set of discrete complex visibilities and grid them incoherently onto a regular UV grid.
    ! In this routine, the weighting function is the Fourier-Transform of a Gaussian beam, with either static
    ! or chromatic beam width.
    ! One can set a parameter ``radmin``, which sets the minimum radius of the window kernel to be applied, in units
    ! of the standard width of the kernel (all frequencies have the same radius, so this sets the radius at the largest
    ! kernel size). This can be safely set to ~5.
    ! ==================================================================================================================

    ! = PARAMETERS =====================================================================================================
    ! == INPUT PARAMETERS ----------------------------------------------------------------------------------------------
    integer, intent(in) :: Nnu                        ! Number of frequency bins
    integer, intent(in) :: Nbl                        ! Number of baselines
    integer, intent(in) :: Ngrid_u, Ngrid_v           ! Number of grid-cells in u,v
    real(8), intent(in) :: f(Nnu)                     ! The frequencies of observation, normalised by a reference frequency
    complex(8), intent(in) :: visibilities(Nbl, Nnu) ! Complex visibilities at each baseline and frequency.
    real(8), intent(in) :: ugrid(Ngrid_u, Ngrid_v)    ! The grid of u.
    real(8), intent(in) :: vgrid(Ngrid_u, Ngrid_v)    ! The grid of v.
    real(8), intent(in) :: baselines(Nbl,2)           ! The UV values of each baseline, at reference frequency
    real(8), intent(in) :: radmin                     ! The minimum radius covered at any frequency, for the beam kernel.
    logical, intent(in) :: static_beam                ! Whether the beam is static or not.
    real(8), intent(in) :: sigma0                     ! The beam width at reference frequency
    integer, intent(in) :: nthreads                   ! NUmber of OMP threads to try to use

    ! -- OUTPUT PARAMETERS ---------------------------------------------------------------------------------------------
    complex(8), intent(out) :: gridded_vis(Ngrid_u,Ngrid_v, Nnu)  ! The gridded visibilities at ugrid, vgrid
    real(8), intent(out)    :: weights(Ngrid_u, Ngrid_v, Nnu)     ! The relative weight of each grid point

    ! -- INTERNAL PARAMETERS -------------------------------------------------------------------------------------------
    integer :: i,j, ncells_u, ncells_v, tot_ncells_u, tot_ncells_v, n_u,n_v, ui0,uif,vi0,vif
    real(8) :: uv(Nbl,2)
    real(8) :: sg(Nnu)
    real(8) :: pi = 3.141592653589
    real(8) :: du,dv, umin,vmin, rmin, umax,vmax
    integer :: u_ind_min, u_ind_max, v_ind_min,v_ind_max
    real(8),allocatable :: u2(:,:), beam(:,:)

    ! ==================================================================================================================

    gridded_vis(:,:,:) = CMPLX(0,0)
    weights(:,:,:) = 0.d0

    ! Get grid dimensions
    du = ugrid(2,1)-ugrid(1,1)
    dv = vgrid(1,2)-vgrid(1,1)
    umin = minval(ugrid)
    vmin = minval(vgrid)
    umax = maxval(ugrid)
    vmax = maxval(vgrid)

    ! Set the beam width
    if(static_beam)then
        sg = sigma0
    else
        sg = sigma0/f
    end if

    ! Get kernel dimensions
    rmin = radmin*maxval(1./(4*pi**2*sg**2))
    ncells_u = int(ceiling(rmin/du))
    ncells_v = int(ceiling(rmin/dv))
    tot_ncells_u = 2*ncells_u + 1
    tot_ncells_v = 2*ncells_v + 1


   !$OMP PARALLEL PRIVATE(i,j,n_u,n_v,ui0,uif,vi0,vif,uv,u_ind_min,&
   !$OMP&           u_ind_max,v_ind_min,v_ind_max,u2,beam)&
   !$OMP& NUM_THREADS(nthreads)

    ! Initial allocation of u2, beam
    allocate(u2(tot_ncells_u,tot_ncells_v), beam(tot_ncells_u, tot_ncells_v))

    !$OMP DO
    do i=1,Nnu
        uv = baselines * f(i)

        do j = 1,Nbl
            ! Check if baseline is "on" the grid
            if ((uv(j,1) + rmin < umin).or.(uv(j,2)+ rmin<vmin) .or. &
                (uv(j,1) - rmin > umax).or.(uv(j,2)- rmin>vmax)) then
                cycle
            end if

            beam(:,:) = 0.d0

            ! Get cell indices at which kernel will apply
            u_ind_min = nint((uv(j,1)- umin)/du) + 1 - ncells_u
            u_ind_max = u_ind_min + tot_ncells_u - 1
            v_ind_min = nint((uv(j,2)- vmin)/dv) + 1 - ncells_u
            v_ind_max = v_ind_min + tot_ncells_v - 1

            !write(*,*) i, uv(i,:), u_ind_min, u_ind_max, v_ind_min, v_ind_max
            ! In case things are off the grid, truncate them.
            ui0 = max(u_ind_min,1)
            uif = min(u_ind_max,Ngrid_u)
            vi0 = max(v_ind_min,1)
            vif = min(v_ind_max,Ngrid_v)


            ! Get number of cells in each direction
            n_u = uif - ui0 + 1
            n_v = vif - vi0 + 1

            ! Check if the beam and kernel co-ordinates are the right size, and re-allocated if necessary.
            !if (size(u2,1)/=n_u .or. size(u2,2)/=n_v) then
            !    deallocate(u2,beam)
            !    allocate(u2(n_u,n_v),beam(n_u,n_v))
            !end if

!            ! Get co-ordinates of kernel
!            if (n_u > tot_ncells_u .or. n_v > tot_ncells_v .or. n_u.eq.1 .or. n_v.eq.1) then
!                write(*,*) "Seeming error: ", n_u, tot_ncells_u, n_v, tot_ncells_v
!            end if

            u2(1:n_u, 1:n_v) = (uv(j,1) - ugrid(ui0:uif,vi0:vif))**2 +&
                               (uv(j,2) - vgrid(ui0:uif,vi0:vif))**2

            ! Get beam kernel
            beam(1:n_u, 1:n_v) = exp(-2*pi**2 * sg(i)**2 * u2(1:n_u, 1:n_v))

            ! Grid the visibility and get weights
            gridded_vis(ui0:uif,vi0:vif,i) = gridded_vis(ui0:uif,vi0:vif,i) + visibilities(j,i)*beam(1:n_u, 1:n_v)
            weights(ui0:uif,vi0:vif,i) = weights(ui0:uif,vi0:vif,i)+beam(1:n_u, 1:n_v)
        end do

    end do
   !$OMP END DO
   !$OMP END PARALLEL

    gridded_vis = gridded_vis/weights


end subroutine


subroutine grid_visibilities_tophat(Nnu,Nbl,Ngrid_u, Ngrid_v, f, visibilities, baselines, ugrid, vgrid, gridded_vis,&
                                    weights,radmin,static_beam, sigma0, q, nthreads )
    implicit none
    ! ==================================================================================================================
    ! grid visibilities
    !
    ! Take a set of discrete complex visibilities and grid them incoherently onto a regular UV grid.
    ! In this routine, the weighting function is a top-hat in u-space.
    ! One can set a parameter ``radmin``, which sets the minimum radius of the window kernel to be applied, in units
    ! of the standard width of the kernel (all frequencies have the same radius, so this sets the radius at the largest
    ! kernel size). This can be safely set to ~5.
    ! ==================================================================================================================

    ! = PARAMETERS =====================================================================================================
    ! == INPUT PARAMETERS ----------------------------------------------------------------------------------------------
    integer, intent(in) :: Nnu                        ! Number of frequency bins
    integer, intent(in) :: Nbl                        ! Number of baselines
    integer, intent(in) :: Ngrid_u, Ngrid_v           ! Number of grid-cells in u,v
    real(8), intent(in) :: f(Nnu)                     ! The frequencies of observation, normalised by a reference frequency
    complex(8), intent(in) :: visibilities(Nbl, Nnu) ! Complex visibilities at each baseline and frequency.
    real(8), intent(in) :: ugrid(Ngrid_u, Ngrid_v)    ! The grid of u.
    real(8), intent(in) :: vgrid(Ngrid_u, Ngrid_v)    ! The grid of v.
    real(8), intent(in) :: baselines(Nbl,2)           ! The UV values of each baseline, at reference frequency
    real(8), intent(in) :: radmin                     ! The minimum radius covered at any frequency, for the beam kernel.
    logical, intent(in) :: static_beam                ! Whether the beam is static or not.
    real(8), intent(in) :: sigma0                     ! The beam width at reference frequency
    integer, intent(in) :: nthreads                   ! NUmber of OMP threads to try to use
    real(8), intent(in) :: q                          ! Radius of the tophat.

    ! -- OUTPUT PARAMETERS ---------------------------------------------------------------------------------------------
    complex(8), intent(out) :: gridded_vis(Ngrid_u,Ngrid_v, Nnu)  ! The gridded visibilities at ugrid, vgrid
    real(8), intent(out)    :: weights(Ngrid_u, Ngrid_v, Nnu)     ! The relative weight of each grid point

    ! -- INTERNAL PARAMETERS -------------------------------------------------------------------------------------------
    integer :: i,j, ncells_u, ncells_v, tot_ncells_u, tot_ncells_v, n_u,n_v, ui0,uif,vi0,vif
    real(8) :: uv(Nbl,2)
    real(8) :: sg(Nnu)
    real(8) :: pi = 3.141592653589
    real(8) :: du,dv, umin,vmin, rmin, umax,vmax
    integer :: u_ind_min, u_ind_max, v_ind_min,v_ind_max
    real(8),allocatable :: u2(:,:), beam(:,:)

    ! ==================================================================================================================

    gridded_vis(:,:,:) = CMPLX(0,0)
    weights(:,:,:) = 0.d0

    ! Get grid dimensions
    du = ugrid(2,1)-ugrid(1,1)
    dv = vgrid(1,2)-vgrid(1,1)
    umin = minval(ugrid)
    vmin = minval(vgrid)
    umax = maxval(ugrid)
    vmax = maxval(vgrid)

    ! Set the beam width
    if(static_beam)then
        sg = sigma0
    else
        sg = sigma0/f
    end if

    ! Get kernel dimensions
    rmin = radmin*maxval(1./(4*pi**2*sg**2))
    ncells_u = int(ceiling(rmin/du))
    ncells_v = int(ceiling(rmin/dv))
    tot_ncells_u = 2*ncells_u + 1
    tot_ncells_v = 2*ncells_v + 1


   !$OMP PARALLEL PRIVATE(i,j,n_u,n_v,ui0,uif,vi0,vif,uv,u_ind_min,&
   !$OMP&           u_ind_max,v_ind_min,v_ind_max,u2,beam)&
   !$OMP& NUM_THREADS(nthreads)

    ! Initial allocation of u2, beam
    allocate(u2(tot_ncells_u,tot_ncells_v), beam(tot_ncells_u, tot_ncells_v))

    !$OMP DO
    do i=1,Nnu
        uv = baselines * f(i)


        do j = 1,Nbl
            ! Check if baseline is "on" the grid
            if ((uv(j,1) + rmin < umin).or.(uv(j,2)+ rmin<vmin) .or. &
                (uv(j,1) - rmin > umax).or.(uv(j,2)- rmin>vmax)) then
                cycle
            end if

            beam(:,:) = 0.d0

            ! Get cell indices at which kernel will apply
            u_ind_min = nint((uv(j,1)- umin)/du) + 1 - ncells_u
            u_ind_max = u_ind_min + tot_ncells_u - 1
            v_ind_min = nint((uv(j,2)- vmin)/dv) + 1 - ncells_u
            v_ind_max = v_ind_min + tot_ncells_v - 1

            !write(*,*) i, uv(i,:), u_ind_min, u_ind_max, v_ind_min, v_ind_max
            ! In case things are off the grid, truncate them.
            ui0 = max(u_ind_min,1)
            uif = min(u_ind_max,Ngrid_u)
            vi0 = max(v_ind_min,1)
            vif = min(v_ind_max,Ngrid_v)


            ! Get number of cells in each direction
            n_u = uif - ui0 + 1
            n_v = vif - vi0 + 1

            ! Check if the beam and kernel co-ordinates are the right size, and re-allocated if necessary.
            !if (size(u2,1)/=n_u .or. size(u2,2)/=n_v) then
            !    deallocate(u2,beam)
            !    allocate(u2(n_u,n_v),beam(n_u,n_v))
            !end if

!            ! Get co-ordinates of kernel
!            if (n_u > tot_ncells_u .or. n_v > tot_ncells_v .or. n_u.eq.1 .or. n_v.eq.1) then
!                write(*,*) "Seeming error: ", n_u, tot_ncells_u, n_v, tot_ncells_v
!            end if

            u2(1:n_u, 1:n_v) = (uv(j,1) - ugrid(ui0:uif,vi0:vif))**2 +&
                               (uv(j,2) - vgrid(ui0:uif,vi0:vif))**2

            ! Get beam kernel
            where (u2(1:n_u, 1:n_v).le.q) beam(1:n_u, 1:n_v) = 1

            ! Grid the visibility and get weights
            gridded_vis(ui0:uif,vi0:vif,i) = gridded_vis(ui0:uif,vi0:vif,i) + visibilities(j,i)*beam(1:n_u, 1:n_v)
            weights(ui0:uif,vi0:vif,i) = weights(ui0:uif,vi0:vif,i)+beam(1:n_u, 1:n_v)
        end do

    end do
   !$OMP END DO
   !$OMP END PARALLEL

    gridded_vis = gridded_vis/weights


end subroutine

!subroutine gridvis_beam_longrange(Nnu,Nbl,Ngrid_u, Ngrid_theta, f, visibilities, baselines, u, theta, gridded_vis,&
!                                  weights,radmin,static_beam, sigma0, eps, nthreads )
!    implicit none
    ! ==================================================================================================================
    ! grid visibilities
    !
    ! Take a set of discrete complex visibilities and grid them incoherently onto a regular UV grid.
    ! In this routine, the weighting function is a top-hat in u-space.
    ! One can set a parameter ``radmin``, which sets the minimum radius of the window kernel to be applied, in units
    ! of the standard width of the kernel (all frequencies have the same radius, so this sets the radius at the largest
    ! kernel size). This can be safely set to ~5.
    ! ==================================================================================================================

    ! = PARAMETERS =====================================================================================================
    ! == INPUT PARAMETERS ----------------------------------------------------------------------------------------------
!    integer, intent(in) :: Nnu                        ! Number of frequency bins
!    integer, intent(in) :: Nbl                        ! Number of baselines
!    integer, intent(in) :: Ngrid_u, Ngrid_theta       ! Number of grid-cells in u,theta
!    real(8), intent(in) :: f(Nnu)                     ! The frequencies of observation, normalised by a reference frequency
!    complex(8), intent(in) :: visibilities(Nbl, Nnu)  ! Complex visibilities at each baseline and frequency.
!    real(8), intent(in) :: u(Ngrid_u)                 ! The grid of u. Arbitrary but increasing.
!    real(8), intent(in) :: theta(Ngrid_theta)         ! The grid of theta.
!    real(8), intent(in) :: baselines(Nbl,2)           ! The UV values of each baseline, at reference frequency. Must be increasing in magnitude.
!    logical, intent(in) :: static_beam                ! Whether the beam is static or not.
!    real(8), intent(in) :: sigma0                     ! The beam width at reference frequency
!    integer, intent(in) :: nthreads                   ! NUmber of OMP threads to try to use
!    real(8), intent(in) :: eps                        ! Determines how far to search for baselines.

    ! -- OUTPUT PARAMETERS ---------------------------------------------------------------------------------------------
 !   complex(8), intent(out) :: gridded_vis(Ngrid_u,Ngrid_theta, Nnu)  ! The gridded visibilities at ugrid, vgrid
 !   real(8), intent(out)    :: weights(Ngrid_u, Ngrid_theta, Nnu)     ! The relative weight of each grid point

    ! -- INTERNAL PARAMETERS -------------------------------------------------------------------------------------------
 !   integer :: i,j, ncells_u, ncells_v, tot_ncells_u, tot_ncells_v, n_u,n_v, ui0,uif,vi0,vif
 !   real(8) :: uv(Nbl,2)
 !   real(8) :: sg(Nnu)
 !   real(8) :: pi = 3.141592653589
 !   real(8) :: du,dv, umin,vmin, rmin, umax,vmax
 !   integer :: u_ind_min, u_ind_max, v_ind_min,v_ind_max
 !   real(8),allocatable :: u2(:,:), beam(:,:)

    ! ==================================================================================================================

!    gridded_vis(:,:,:) = CMPLX(0,0)
!    weights(:,:,:) = 0.d0

    ! Get grid dimensions
 !   du = ugrid(2,1)-ugrid(1,1)
 !   dv = vgrid(1,2)-vgrid(1,1)
 !   umin = minval(ugrid)
 !   vmin = minval(vgrid)
 !   umax = maxval(ugrid)
 !   vmax = maxval(vgrid)

    ! Set the beam width
 !   if(static_beam)then
 !       sg = sigma0
 !   else
 !       sg = sigma0/f
  !  end if


 !  !$OMP PARALLEL PRIVATE(i,j,n_u,n_v,ui0,uif,vi0,vif,uv,u_ind_min,&
 !  !$OMP&           u_ind_max,v_ind_min,v_ind_max,u2,beam)&
 !  !$OMP& NUM_THREADS(nthreads)

!    ! Initial allocation of u2, beam
!    allocate(u2(tot_ncells_u,tot_ncells_v), beam(tot_ncells_u, tot_ncells_v))

 !   !$OMP DO
  !  do i=1,Nnu
  !      uv = baselines * f(i)


!        do j = 1,Ngrid_u


 !           do k=1, Ngrid_theta
 !               ! Check if baseline is "on" the grid
 !               if ((uv(j,1) + rmin < umin).or.(uv(j,2)+ rmin<vmin) .or. &
 !                   (uv(j,1) - rmin > umax).or.(uv(j,2)- rmin>vmax)) then
 !                   cycle
 !               end if

 !               beam(:,:) = 0.d0

                ! Get cell indices at which kernel will apply
 !               u_ind_min = nint((uv(j,1)- umin)/du) + 1 - ncells_u
 !               u_ind_max = u_ind_min + tot_ncells_u - 1
 !               v_ind_min = nint((uv(j,2)- vmin)/dv) + 1 - ncells_u
 !               v_ind_max = v_ind_min + tot_ncells_v - 1

                !write(*,*) i, uv(i,:), u_ind_min, u_ind_max, v_ind_min, v_ind_max
                ! In case things are off the grid, truncate them.
!                ui0 = max(u_ind_min,1)
!                uif = min(u_ind_max,Ngrid_u)
!                vi0 = max(v_ind_min,1)
!                vif = min(v_ind_max,Ngrid_v)


                ! Get number of cells in each direction
!                n_u = uif - ui0 + 1
!                n_v = vif - vi0 + 1

                ! Check if the beam and kernel co-ordinates are the right size, and re-allocated if necessary.
                !if (size(u2,1)/=n_u .or. size(u2,2)/=n_v) then
                !    deallocate(u2,beam)
                !    allocate(u2(n_u,n_v),beam(n_u,n_v))
                !end if

    !            ! Get co-ordinates of kernel
    !            if (n_u > tot_ncells_u .or. n_v > tot_ncells_v .or. n_u.eq.1 .or. n_v.eq.1) then
    !                write(*,*) "Seeming error: ", n_u, tot_ncells_u, n_v, tot_ncells_v
    !            end if

!                u2(1:n_u, 1:n_v) = (uv(j,1) - ugrid(ui0:uif,vi0:vif))**2 +&
 !                                  (uv(j,2) - vgrid(ui0:uif,vi0:vif))**2

                ! Get beam kernel
 !               where (u2(1:n_u, 1:n_v).le.q) beam(1:n_u, 1:n_v) = 1

                ! Grid the visibility and get weights
 !               gridded_vis(ui0:uif,vi0:vif,i) = gridded_vis(ui0:uif,vi0:vif,i) + visibilities(j,i)*beam(1:n_u, 1:n_v)
 !               weights(ui0:uif,vi0:vif,i) = weights(ui0:uif,vi0:vif,i)+beam(1:n_u, 1:n_v)

 !           end do
 !       end do

 !   end do
!   !$OMP END DO
!   !$OMP END PARALLEL

 !   gridded_vis = gridded_vis/weights


!end subroutine
