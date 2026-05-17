`default_nettype none

module ready_valid_sum (
    input  wire              clk,
    input  wire              rst,
    input  wire              input_valid,
    output wire              input_ready,
    input  wire              input_last,
    input  wire signed [7:0] input_value,
    output reg               result_valid,
    input  wire              result_ready,
    output reg signed [15:0] result_value
);
    reg signed [15:0] accumulator;

    assign input_ready = !result_valid;

    always_ff @(posedge clk) begin
        if (rst) begin
            accumulator <= 16'sd0;
            result_valid <= 1'b0;
            result_value <= 16'sd0;
        end else if (result_valid) begin
            if (result_ready) begin
                result_valid <= 1'b0;
            end
        end else if (input_valid && input_ready) begin
            if (input_last) begin
                result_value <= accumulator + input_value;
                accumulator <= 16'sd0;
                result_valid <= 1'b1;
            end else begin
                accumulator <= accumulator + input_value;
            end
        end
    end
endmodule

`default_nettype wire
