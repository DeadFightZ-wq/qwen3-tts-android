package com.qwen.tts.android.data

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteOrder
import kotlin.math.floor
import kotlin.math.roundToInt

internal data class NormalizedVoiceAudio(
    val file: File,
    val durationMillis: Long,
)

/**
 * Decodes a local audio document and writes the format consumed by the native
 * Qwen speaker-embedding extractor: mono, 24 kHz, signed PCM 16-bit WAV.
 *
 * MediaExtractor/MediaCodec are used intentionally so decoding remains on-device
 * and does not add a third-party native dependency to the app.
 */
internal object AudioNormalizer {
    private const val TARGET_SAMPLE_RATE = 24_000
    private const val MAX_DURATION_SECONDS = 120
    private const val CODEC_TIMEOUT_US = 10_000L
    private const val MAX_IDLE_OUTPUT_ATTEMPTS = 1_000

    fun normalize(context: Context, uri: Uri, outputFile: File): NormalizedVoiceAudio {
        val extractor = MediaExtractor()
        var decoder: MediaCodec? = null
        var decoderStarted = false

        try {
            extractor.setDataSource(context, uri, null)

            val audioTrackIndex = (0 until extractor.trackCount).firstOrNull { index ->
                val format = extractor.getTrackFormat(index)
                format.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
            } ?: error("The selected file does not contain an audio track.")

            val inputFormat = extractor.getTrackFormat(audioTrackIndex)
            val mime = inputFormat.getString(MediaFormat.KEY_MIME)
                ?: error("The selected audio format could not be identified.")

            if (inputFormat.containsKey(MediaFormat.KEY_DURATION)) {
                val durationUs = inputFormat.getLong(MediaFormat.KEY_DURATION)
                if (durationUs > MAX_DURATION_SECONDS * 1_000_000L) {
                    error("The reference audio must be ${MAX_DURATION_SECONDS} seconds or shorter.")
                }
            }

            extractor.selectTrack(audioTrackIndex)
            decoder = MediaCodec.createDecoderByType(mime)
            decoder.configure(inputFormat, null, null, 0)
            decoder.start()
            decoderStarted = true

            var sampleRate = inputFormat.getIntegerOrDefault(MediaFormat.KEY_SAMPLE_RATE, 0)
            var channelCount = inputFormat.getIntegerOrDefault(MediaFormat.KEY_CHANNEL_COUNT, 0)
            var pcmEncoding = AudioFormat.ENCODING_PCM_16BIT
            val monoSamples = FloatArrayBuilder()

            var inputEnded = false
            var outputEnded = false
            var idleOutputAttempts = 0
            val bufferInfo = MediaCodec.BufferInfo()

            while (!outputEnded) {
                if (!inputEnded) {
                    val inputIndex = decoder.dequeueInputBuffer(CODEC_TIMEOUT_US)
                    if (inputIndex >= 0) {
                        val inputBuffer = decoder.getInputBuffer(inputIndex)
                            ?: error("The audio decoder did not provide an input buffer.")
                        inputBuffer.clear()

                        val sampleSize = extractor.readSampleData(inputBuffer, 0)
                        if (sampleSize < 0) {
                            decoder.queueInputBuffer(
                                inputIndex,
                                0,
                                0,
                                0L,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                            )
                            inputEnded = true
                        } else {
                            val presentationTimeUs = extractor.sampleTime.coerceAtLeast(0L)
                            decoder.queueInputBuffer(
                                inputIndex,
                                0,
                                sampleSize,
                                presentationTimeUs,
                                0,
                            )
                            extractor.advance()
                        }
                    }
                }

                when (val outputIndex = decoder.dequeueOutputBuffer(bufferInfo, CODEC_TIMEOUT_US)) {
                    MediaCodec.INFO_TRY_AGAIN_LATER -> {
                        if (inputEnded) {
                            idleOutputAttempts += 1
                            if (idleOutputAttempts > MAX_IDLE_OUTPUT_ATTEMPTS) {
                                error("The audio decoder stopped before finishing the file.")
                            }
                        }
                    }

                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val outputFormat = decoder.outputFormat
                        sampleRate = outputFormat.getIntegerOrDefault(MediaFormat.KEY_SAMPLE_RATE, sampleRate)
                        channelCount = outputFormat.getIntegerOrDefault(MediaFormat.KEY_CHANNEL_COUNT, channelCount)
                        pcmEncoding = outputFormat.getIntegerOrDefault(
                            MediaFormat.KEY_PCM_ENCODING,
                            AudioFormat.ENCODING_PCM_16BIT,
                        )
                        validateOutputFormat(sampleRate, channelCount, pcmEncoding)
                    }

                    MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> Unit

                    else -> if (outputIndex >= 0) {
                        idleOutputAttempts = 0
                        if (bufferInfo.size > 0) {
                            if (sampleRate <= 0 || channelCount <= 0) {
                                val outputFormat = decoder.outputFormat
                                sampleRate = outputFormat.getIntegerOrDefault(MediaFormat.KEY_SAMPLE_RATE, sampleRate)
                                channelCount = outputFormat.getIntegerOrDefault(MediaFormat.KEY_CHANNEL_COUNT, channelCount)
                                pcmEncoding = outputFormat.getIntegerOrDefault(
                                    MediaFormat.KEY_PCM_ENCODING,
                                    AudioFormat.ENCODING_PCM_16BIT,
                                )
                                validateOutputFormat(sampleRate, channelCount, pcmEncoding)
                            }

                            val outputBuffer = decoder.getOutputBuffer(outputIndex)
                                ?: error("The audio decoder returned an empty output buffer.")
                            val pcm = outputBuffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
                            pcm.position(bufferInfo.offset)
                            pcm.limit(bufferInfo.offset + bufferInfo.size)

                            appendMonoPcm(
                                pcm = pcm.slice().order(ByteOrder.LITTLE_ENDIAN),
                                channelCount = channelCount,
                                encoding = pcmEncoding,
                                sampleRate = sampleRate,
                                destination = monoSamples,
                            )
                        }

                        val reachedEnd = (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                        decoder.releaseOutputBuffer(outputIndex, false)
                        if (reachedEnd) outputEnded = true
                    }
                }
            }

            if (sampleRate <= 0 || monoSamples.size == 0) {
                error("No decodable audio samples were found in the selected file.")
            }

            val normalizedSamples = resample(monoSamples.toArray(), sampleRate, TARGET_SAMPLE_RATE)
            if (normalizedSamples.isEmpty()) {
                error("The selected audio does not contain enough samples to create a voice profile.")
            }

            outputFile.parentFile?.mkdirs()
            writeWav(outputFile, normalizedSamples, TARGET_SAMPLE_RATE)

            return NormalizedVoiceAudio(
                file = outputFile,
                durationMillis = normalizedSamples.size * 1_000L / TARGET_SAMPLE_RATE,
            )
        } catch (throwable: Throwable) {
            runCatching { outputFile.delete() }
            throw throwable
        } finally {
            if (decoderStarted) {
                runCatching { decoder?.stop() }
            }
            runCatching { decoder?.release() }
            runCatching { extractor.release() }
        }
    }

    private fun validateOutputFormat(sampleRate: Int, channels: Int, encoding: Int) {
        if (sampleRate !in 8_000..192_000) {
            error("The audio sample rate ($sampleRate Hz) is not supported.")
        }
        if (channels !in 1..8) {
            error("The audio channel layout ($channels channels) is not supported.")
        }
        if (bytesPerSample(encoding) == 0) {
            error("The audio decoder returned an unsupported PCM encoding.")
        }
    }

    private fun appendMonoPcm(
        pcm: java.nio.ByteBuffer,
        channelCount: Int,
        encoding: Int,
        sampleRate: Int,
        destination: FloatArrayBuilder,
    ) {
        val sampleWidth = bytesPerSample(encoding)
        if (sampleWidth == 0) error("The decoded audio uses an unsupported PCM encoding.")

        val frameSize = sampleWidth * channelCount
        val frameCount = pcm.remaining() / frameSize
        val maximumFrames = sampleRate.toLong() * MAX_DURATION_SECONDS
        if (destination.size.toLong() + frameCount > maximumFrames) {
            error("The reference audio must be ${MAX_DURATION_SECONDS} seconds or shorter.")
        }

        repeat(frameCount) {
            var sum = 0f
            repeat(channelCount) {
                sum += readPcmSample(pcm, encoding)
            }
            destination.add((sum / channelCount).coerceIn(-1f, 1f))
        }
    }

    private fun bytesPerSample(encoding: Int): Int = when (encoding) {
        AudioFormat.ENCODING_PCM_8BIT -> 1
        AudioFormat.ENCODING_PCM_16BIT -> 2
        AudioFormat.ENCODING_PCM_FLOAT,
        AudioFormat.ENCODING_PCM_32BIT -> 4
        AudioFormat.ENCODING_PCM_24BIT_PACKED -> 3
        else -> 0
    }

    private fun readPcmSample(buffer: java.nio.ByteBuffer, encoding: Int): Float = when (encoding) {
        AudioFormat.ENCODING_PCM_8BIT -> ((buffer.get().toInt() and 0xff) - 128) / 128f
        AudioFormat.ENCODING_PCM_16BIT -> buffer.short / 32768f
        AudioFormat.ENCODING_PCM_FLOAT -> buffer.float.coerceIn(-1f, 1f)
        AudioFormat.ENCODING_PCM_24BIT_PACKED -> {
            val b0 = buffer.get().toInt() and 0xff
            val b1 = buffer.get().toInt() and 0xff
            val b2 = buffer.get().toInt() and 0xff
            var value = (b2 shl 16) or (b1 shl 8) or b0
            if ((value and 0x800000) != 0) value = value or -0x1000000
            value / 8_388_608f
        }
        AudioFormat.ENCODING_PCM_32BIT -> (buffer.int / 2_147_483_648.0).toFloat()
        else -> error("Unsupported PCM encoding: $encoding")
    }

    private fun resample(input: FloatArray, inputRate: Int, outputRate: Int): FloatArray {
        if (inputRate == outputRate) return input

        val outputCount = (input.size.toDouble() * outputRate / inputRate).roundToInt()
        if (outputCount <= 0) return FloatArray(0)

        return FloatArray(outputCount) { outputIndex ->
            val sourcePosition = outputIndex.toDouble() * inputRate / outputRate
            val leftIndex = floor(sourcePosition).toInt().coerceIn(0, input.lastIndex)
            val rightIndex = (leftIndex + 1).coerceAtMost(input.lastIndex)
            val fraction = (sourcePosition - leftIndex).toFloat()
            input[leftIndex] + (input[rightIndex] - input[leftIndex]) * fraction
        }
    }

    private fun writeWav(file: File, samples: FloatArray, sampleRate: Int) {
        val dataSize = samples.size * 2
        BufferedOutputStream(FileOutputStream(file)).use { output ->
            output.writeAscii("RIFF")
            output.writeIntLe(36 + dataSize)
            output.writeAscii("WAVE")
            output.writeAscii("fmt ")
            output.writeIntLe(16)
            output.writeShortLe(1)
            output.writeShortLe(1)
            output.writeIntLe(sampleRate)
            output.writeIntLe(sampleRate * 2)
            output.writeShortLe(2)
            output.writeShortLe(16)
            output.writeAscii("data")
            output.writeIntLe(dataSize)

            samples.forEach { sample ->
                val pcmValue = (sample.coerceIn(-1f, 1f) * 32767f).roundToInt().coerceIn(-32768, 32767)
                output.write(pcmValue and 0xff)
                output.write((pcmValue shr 8) and 0xff)
            }
        }
    }

    private fun MediaFormat.getIntegerOrDefault(key: String, default: Int): Int =
        if (containsKey(key)) getInteger(key) else default

    private fun java.io.OutputStream.writeAscii(value: String) {
        write(value.toByteArray(Charsets.US_ASCII))
    }

    private fun java.io.OutputStream.writeIntLe(value: Int) {
        write(value and 0xff)
        write((value shr 8) and 0xff)
        write((value shr 16) and 0xff)
        write((value shr 24) and 0xff)
    }

    private fun java.io.OutputStream.writeShortLe(value: Int) {
        write(value and 0xff)
        write((value shr 8) and 0xff)
    }

    private class FloatArrayBuilder {
        private var values = FloatArray(16_384)
        var size: Int = 0
            private set

        fun add(value: Float) {
            if (size == values.size) {
                val nextSize = (values.size * 2).coerceAtMost(25_000_000)
                if (nextSize <= values.size) {
                    error("The decoded reference audio is too large to process.")
                }
                values = values.copyOf(nextSize)
            }
            values[size++] = value
        }

        fun toArray(): FloatArray = values.copyOf(size)
    }
}
