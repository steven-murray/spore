subroutine direct_ft(N_u, N_pos, f0, u0, v0, pos, visible_flux, visibility)
	implicit none

	integer, intent(in) :: N_u, N_pos
	real(8), intent(in) :: f0, u0(N_u), v0(N_u), pos(2,N_pos), visible_flux(N_pos)

	
	real(8) :: twopi = 6.28318530718
	integer :: i

	complex(8), intent(out) :: visibility(N_u)
	

	!$OMP PARALLEL DO
	do i=1,N_u
		visibility(i) = sum(visible_flux * exp(-CMPLX(0, twopi*f0*(u0(i)*pos(1,:)+v0(i)*pos(2,:)), kind=8)))
	end do
	!$OMP END PARALLEL DO


end subroutine
