c     calculating height h ====================================================
      subroutine calc_h(uvh, h, Nx, Ny, H0, Ih_i)
      implicit none

      integer Nx, Ny
      double precision uvh(3*Ny, Nx), h(Ny, Nx)

      double precision H0
      integer Ih_i, i, j

cf2py intent(in) :: uvh, H0, Ih_i
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: h

      do j=1,Nx
        do i=1,Ny
          h(i, j) = H0 + uvh(i+Ih_i, j)
        enddo
      enddo

      end


c     calculating zonal mass flux U ===========================================
      subroutine calc_U(uvh, h, U, Nx, Ny)
      implicit none

      integer Nx, Ny
      double precision uvh(3*Ny, Nx), h(Ny, Nx), U(Ny, Nx)

      integer i, j

cf2py intent(in) :: uvh, h
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: U

      do j=1,Nx-1
        do i=1,Ny
          U(i, j) = 0.5*(h(i, j+1) + h(i,j))  * uvh(i, j)
          !            roll
        enddo
      enddo

      do i=1,Ny
        U(i, Nx) = 0.5*(h(i, 1) + h(i, Nx)) * uvh(i, Nx)
      enddo

      end


c     calculating meridonal mass flux V =======================================
      subroutine calc_V(uvh, h, V, Nx, Ny, Iv_i)
      implicit none

      integer Nx, Ny
      double precision uvh(3*Ny, Nx), h(Ny, Nx), V(Ny, Nx)

      integer Iv_i, i, j

cf2py intent(in) :: uvh, h, Iv_i
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: V

      do j=1,Nx
        do i=1,Ny-1
          V(i, j) = 0.5*(h(i+1, j) + h(i,j)) * uvh(i+Iv_i, j)
        enddo

        V(Ny, j) = 0.5*(h(1, j) + h(Ny, j)) * uvh(Ny+Iv_i, j)
      enddo


      end