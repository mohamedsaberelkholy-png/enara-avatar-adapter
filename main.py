/**
 * Enara Avatar Adapter - Complete Testing Script
 * Tests both session key fix and Arabic detection
 */

const ADAPTER_URL = 'https://enara-avatar-adapter-production.up.railway.app';
const AUTH_TOKEN = 'EnaraAvatar2026!';
const POLL_INTERVAL = 2000; // 2 seconds
const MAX_POLLS = 30; // 60 seconds total

/**
 * Test 1: English Question (Should Generate Visual)
 */
async function testEnglish() {
  console.log('\n========== TEST 1: ENGLISH (WITH VISUAL) ==========\n');
  
  const sessionId = `test-eng-${Date.now()}`;
  console.log(`Session ID: ${sessionId}\n`);

  try {
    // 1. Send chat request
    console.log('1️⃣ Sending English question...');
    const chatRes = await fetch(`${ADAPTER_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${AUTH_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'enara-tutor',
        messages: [
          { 
            role: 'system', 
            content: `You are Enara, an AI tutor.\nSession: \nSession: ${sessionId}` 
          },
          { 
            role: 'user', 
            content: 'What is the difference between present simple and present continuous? Please give me a table.' 
          }
        ]
      })
    });

    if (!chatRes.ok) {
      throw new Error(`Chat request failed: ${chatRes.status} ${chatRes.statusText}`);
    }

    // Stream the response
    const reader = chatRes.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    
    console.log('2️⃣ Streaming response:');
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      process.stdout.write(chunk);
      fullText += chunk;
    }
    console.log('\n');

    // 2. Wait for visual generation
    console.log('3️⃣ Waiting for visual generation (5 seconds)...');
    await new Promise(r => setTimeout(r, 5000));

    // 3. Poll for visual artifact
    console.log(`4️⃣ Polling for visual artifact (${POLL_INTERVAL}ms intervals)...\n`);
    await pollForArtifact(sessionId);

  } catch (err) {
    console.error('❌ Test 1 Error:', err.message);
  }
}

/**
 * Test 2: Native Arabic (Should Detect & Generate Visual)
 */
async function testArabic() {
  console.log('\n========== TEST 2: NATIVE ARABIC ==========\n');
  
  const sessionId = `test-ar-${Date.now()}`;
  console.log(`Session ID: ${sessionId}\n`);

  try {
    console.log('1️⃣ Sending Arabic question (native script)...');
    const chatRes = await fetch(`${ADAPTER_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${AUTH_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'enara-tutor',
        messages: [
          { 
            role: 'system', 
            content: `You are Enara, an AI tutor.\nSession: \nSession: ${sessionId}` 
          },
          { 
            role: 'user', 
            content: 'ما الفرق بين المضارع البسيط والمضارع المستمر؟ أعطني جدول' 
          }
        ]
      })
    });

    if (!chatRes.ok) {
      throw new Error(`Chat request failed: ${chatRes.status} ${chatRes.statusText}`);
    }

    const reader = chatRes.body.getReader();
    const decoder = new TextDecoder();
    
    console.log('2️⃣ Streaming response:');
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      process.stdout.write(decoder.decode(value));
    }
    console.log('\n');

    console.log('3️⃣ Waiting for visual generation (5 seconds)...');
    await new Promise(r => setTimeout(r, 5000));

    console.log(`4️⃣ Polling for visual artifact...\n`);
    await pollForArtifact(sessionId);

  } catch (err) {
    console.error('❌ Test 2 Error:', err.message);
  }
}

/**
 * Test 3: Romanized Arabic (Should Detect via New Word List)
 */
async function testRomanizedArabic() {
  console.log('\n========== TEST 3: ROMANIZED ARABIC ==========\n');
  
  const sessionId = `test-rom-${Date.now()}`;
  console.log(`Session ID: ${sessionId}\n`);

  try {
    console.log('1️⃣ Sending romanized Arabic question...');
    const chatRes = await fetch(`${ADAPTER_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${AUTH_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'enara-tutor',
        messages: [
          { 
            role: 'system', 
            content: `You are Enara, an AI tutor.\nSession: \nSession: ${sessionId}` 
          },
          { 
            role: 'user', 
            content: 'Marhaba, shu hada? Yalla, inshallah explain the difference between present and past tense' 
          }
        ]
      })
    });

    if (!chatRes.ok) {
      throw new Error(`Chat request failed: ${chatRes.status} ${chatRes.statusText}`);
    }

    const reader = chatRes.body.getReader();
    const decoder = new TextDecoder();
    
    console.log('2️⃣ Streaming response:');
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      process.stdout.write(decoder.decode(value));
    }
    console.log('\n');

    console.log('3️⃣ Waiting for visual generation (5 seconds)...');
    await new Promise(r => setTimeout(r, 5000));

    console.log(`4️⃣ Polling for visual artifact...\n`);
    await pollForArtifact(sessionId);

  } catch (err) {
    console.error('❌ Test 3 Error:', err.message);
  }
}

/**
 * Helper: Poll for artifact until found or timeout
 */
async function pollForArtifact(sessionId, pollCount = 0) {
  try {
    const res = await fetch(`${ADAPTER_URL}/v1/artifact/${sessionId}`);
    const data = await res.json();

    if (data.html) {
      console.log(`✅ VISUAL FOUND! (Poll #${pollCount})`);
      console.log(`   Session Key: ${data.session_key}`);
      console.log(`   HTML Length: ${data.html.length} chars`);
      console.log(`   HTML Preview: ${data.html.substring(0, 120)}...\n`);
      return true;
    } else {
      if (pollCount < MAX_POLLS) {
        process.stdout.write(`.`);
        await new Promise(r => setTimeout(r, POLL_INTERVAL));
        return pollForArtifact(sessionId, pollCount + 1);
      } else {
        console.log(`\n❌ TIMEOUT: No visual after ${MAX_POLLS} polls (${MAX_POLLS * POLL_INTERVAL / 1000}s)`);
        return false;
      }
    }
  } catch (err) {
    console.error(`\n❌ Artifact poll error: ${err.message}`);
    return false;
  }
}

/**
 * Run all tests
 */
async function runAllTests() {
  console.log('╔════════════════════════════════════════════════╗');
  console.log('║  Enara Avatar Adapter - Complete Test Suite   ║');
  console.log('║  Tests: Session Key Fix + Arabic Detection     ║');
  console.log('╚════════════════════════════════════════════════╝');

  await testEnglish();
  await testArabic();
  await testRomanizedArabic();

  console.log('\n╔════════════════════════════════════════════════╗');
  console.log('║           All Tests Completed                  ║');
  console.log('╚════════════════════════════════════════════════╝\n');
}

// Run
runAllTests().catch(console.error);
