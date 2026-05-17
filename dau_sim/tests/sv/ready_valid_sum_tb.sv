`timescale 1ns/1ps
`default_nettype none

module ready_valid_sum_tb;
    logic clk = 1'b0;
    logic rst = 1'b1;
    logic input_valid = 1'b0;
    logic input_ready;
    logic input_last = 1'b0;
    logic signed [7:0] input_value = 8'sd0;
    logic result_valid;
    logic result_ready = 1'b0;
    logic signed [15:0] result_value;

    ready_valid_sum dut (
        .clk(clk),
        .rst(rst),
        .input_valid(input_valid),
        .input_ready(input_ready),
        .input_last(input_last),
        .input_value(input_value),
        .result_valid(result_valid),
        .result_ready(result_ready),
        .result_value(result_value)
    );

    always #5 clk = ~clk;

    task automatic tick;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task automatic apply_reset;
        begin
            rst = 1'b1;
            input_valid = 1'b0;
            input_last = 1'b0;
            result_ready = 1'b0;
            tick();
            rst = 1'b0;
            tick();
        end
    endtask

    task automatic send_sample(input logic signed [7:0] value, input logic last_sample);
        begin
            if (input_ready !== 1'b1) begin
                $fatal(1, "input_ready was low before sample");
            end
            input_value = value;
            input_last = last_sample;
            input_valid = 1'b1;
            tick();
            input_valid = 1'b0;
            input_last = 1'b0;
        end
    endtask

    initial begin
        apply_reset();

        send_sample(8'sd10, 1'b0);
        send_sample(-8'sd3, 1'b0);
        send_sample(8'sd5, 1'b1);
        if (result_valid !== 1'b1 || result_value !== 16'sd12) begin
            $fatal(1, "sum result mismatch: valid=%0b value=%0d", result_valid, result_value);
        end
        if (input_ready !== 1'b0) begin
            $fatal(1, "input_ready did not deassert while result was held");
        end

        input_valid = 1'b1;
        input_last = 1'b1;
        input_value = 8'sd99;
        tick();
        if (result_valid !== 1'b1 || result_value !== 16'sd12) begin
            $fatal(1, "held result changed under backpressure");
        end

        input_valid = 1'b0;
        input_last = 1'b0;
        result_ready = 1'b1;
        tick();
        result_ready = 1'b0;
        if (result_valid !== 1'b0) begin
            $fatal(1, "result_valid did not clear after result_ready");
        end

        send_sample(8'sd4, 1'b1);
        if (result_valid !== 1'b1 || result_value !== 16'sd4) begin
            $fatal(1, "single-sample result mismatch");
        end

        $display("READY_VALID_SUM_TB_OK");
        $finish;
    end
endmodule

`default_nettype wire
