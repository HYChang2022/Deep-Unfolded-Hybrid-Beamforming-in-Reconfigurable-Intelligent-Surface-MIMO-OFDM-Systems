function [x_hat]=VBLAST_decoder(y,Nt,R)
for i = Nt: -1 : 1
    temp = y(i);
    for j = i+1 : Nt
        temp = temp - R(i,j)*x_hat(j);
    end
    x_sliced = temp / R(i,i);
    x_hat(i) = QAM16_slicer(x_sliced,1);
end